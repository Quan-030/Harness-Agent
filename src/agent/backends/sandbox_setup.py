# src/backends/sandbox_setup.py
"""
OpenSandbox 沙箱的初始化与文件播种模块。

职责:
1. 获取或创建 OpenSandbox 沙箱，包装为 OpenSandboxBackend。
2. 播种技能文件（技能包 SKILL.md）。

注意：AGENTS.md 已迁移到 StoreBackend（全局共享），不经过沙箱。
运行时的增量技能同步由 SkillsSyncMiddleware 负责。
（Memory v2 已移除 /memories/ Markdown 记忆路径，方案 5.7）
"""
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

from opensandbox import SandboxSync

from agent.backends.custom_opensandbox import OpenSandboxBackend
from agent.config import (
    LOCAL_SKILLS_DIR, SANDBOX_SKILLS_ROOT,
)


def cleanup_legacy_memories(backend: OpenSandboxBackend) -> None:
    """物理清理沙箱内旧 Markdown 记忆目录（方案 5.7/21.1，source-of-truth 边界）。

    - 未命中 CompositeBackend route 的路径会落到 default sandbox backend，
      且 execute 可直接访问沙箱文件系统——必须物理删除 /memories 才能保证
      MySQL 是唯一运行时事实源（历史数据由部署前离线归档，MongoDB Store 保留）。
    - fail closed：backend.execute 不因非 0 退出码抛异常（返回 ExecuteResponse），
      必须显式检查 exit_code；删除后再 verify 目录不存在。失败则中止沙箱初始化，
      绝不把仍可读取旧记忆的沙箱交给 Agent。
    """
    result = backend.execute("rm -rf /memories")
    if result.exit_code != 0:
        raise RuntimeError(
            f"legacy memory cleanup failed (exit_code={result.exit_code})"
        )
    verify = backend.execute("test ! -e /memories")
    if verify.exit_code != 0:
        raise RuntimeError("legacy memory directory still exists")
    print("[INFO] 已清理沙箱旧 /memories 目录（Memory v2：旧 Markdown 记忆废弃）")


def setup_sandbox(config, sandbox_id=None, image=None) -> OpenSandboxBackend:
    """
    获取或创建沙箱，播种基础文件。

    Args:
        config: ConnectionConfigSync 配置。
        sandbox_id: 可选，要连接的现有沙箱 ID。
        image: 可选，创建新沙箱时使用的镜像。

    Returns:
        OpenSandboxBackend 实例。
    """
    if sandbox_id:
        print(f"[INFO] 正在连接到现有沙箱: {sandbox_id}")
        try:
            sandbox = SandboxSync.connect(sandbox_id, connection_config=config)
            print(f"[INFO] 成功连接到沙箱: {sandbox_id}")
        except Exception as e:
            print(f"[WARNING] 连接沙箱失败: {e}，将创建新沙箱")
            sandbox_id = None

    if not sandbox_id:
        if not image:
            image = "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.2"

        print(f"[INFO] 正在创建新沙箱，使用镜像: {image}")
        sandbox = SandboxSync.create(
            image,
            entrypoint=["/opt/opensandbox/code-interpreter.sh"],
            env={"PYTHON_VERSION": "3.11"},
            resource={"cpu": "4", "memory": "8Gi"},
            timeout=timedelta(hours=2),
            connection_config=config,
            # network_policy=NetworkPolicy(  # 沙箱网络路由限制策略
            #     defaultAction="deny",
            #     egress=[
            #         NetworkRule(action="allow", target="pypi.org"),
            #         NetworkRule(action="allow", target="*.github.com"),
            #     ]
            # )
        )

    backend = OpenSandboxBackend(sandbox=sandbox)
    print(f"[INFO] 沙箱就绪，ID: {sandbox.id}")

    # Memory v2（方案 5.7/21.1）：fail closed 清理旧 Markdown 记忆目录
    # （清理失败不得返回可用沙箱——旧记忆仍可经 default backend/execute 读取）
    cleanup_legacy_memories(backend)

    # 预创建 skills 需要的目录，避免 Agent 运行时遇到 FileNotFoundError
    _ensure_dirs(backend)

    # 播种基础文件（AGENTS.md、Skills）
    _seed_files(backend)

    # 创建 Python venv + 预装第三方依赖
    _create_venv(backend)

    return backend


# skills 运行时依赖的目录（需在沙箱中预创建）
_SKILL_DIRS = ["/analysis/temp"]

# 所有 Python 依赖统一安装到此 venv，避开系统 Python 的 externally managed 限制
_VENV_PATH = "/opt/skills-venv"
_VENV_PIP = f"{_VENV_PATH}/bin/pip"
# skills 运行时需要的 Python 第三方包
_PREINSTALL_PACKAGES = ["numpy", "pandas", "matplotlib", "requests", "beautifulsoup4"]


def _ensure_dirs(backend: OpenSandboxBackend) -> None:
    """预创建 skills 运行所需的目录，避免 FileNotFoundError。"""
    for d in _SKILL_DIRS:
        backend.execute(f"mkdir -p {d}")


# 阿里云 PyPI 镜像，沙箱内走内网加速
_PYPI_INDEX = "https://mirrors.aliyun.com/pypi/simple/"

# 清华 PyPI 镜像，沙箱内走内网加速
# _PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 中科大 PyPI 镜像，沙箱内走内网加速
# _PYPI_INDEX = "https://pypi.mirrors.ustc.edu.cn/simple/"


# pip install 通用参数（default-timeout=300 避免大包下载超时）
_PIP_INSTALL_ARGS = f"-i {_PYPI_INDEX} --default-timeout=300 --no-input -q"


def _create_venv(backend: OpenSandboxBackend) -> None:
    """创建沙箱级 Python venv，并预装 skills 所需的第三方包。

    系统 Python 设置了 externally managed 限制（PEP 668），--system 安装会被拒绝。
    因此创建一个统一的 venv，并将 /opt/skills-venv/bin 注入 SANDBOX_PATH 最前面，
    所有 skill 的 python/pip 命令自动路由到 venv，无需改任何脚本。
    """
    # 1. 创建 venv（幂等：已存在则跳过）
    result = backend.execute(f"python3 -m venv {_VENV_PATH}")
    if result.exit_code != 0:
        print(f"[WARNING] venv 创建失败: {result.output[:200]}")
        return
    print(f"[INFO] Python venv 就绪: {_VENV_PATH}")

    # 2. 升级 pip（镜像加速，60s 超时）
    backend.execute(f"{_VENV_PIP} install --upgrade pip {_PIP_INSTALL_ARGS}", timeout=300)

    # 3. 预装依赖（sentinel 避免重复安装，合并为一条命令减少 HTTP 往返）
    missing_packages = []
    for pkg in _PREINSTALL_PACKAGES:
        sentinel = f"/tmp/.venv_installed_{pkg}"
        check = backend.execute(f"test -f {sentinel}")
        if check.exit_code != 0:
            missing_packages.append(pkg)

    if missing_packages:
        packages_str = " ".join(missing_packages)
        print(f"[INFO] 正在安装 Python 依赖: {packages_str}...")
        result = backend.execute(
            f"{_VENV_PIP} install {packages_str} {_PIP_INSTALL_ARGS}",
            timeout=600,  # 合并安装，给 10 分钟
        )
        if result.exit_code == 0:
            for pkg in missing_packages:
                backend.execute(f"touch /tmp/.venv_installed_{pkg}")
            print(f"[INFO]   所有依赖安装成功 ({len(missing_packages)} 个包)")
        else:
            print(f"[WARNING] 依赖安装失败: {result.output[:200]}")
    else:
        print("[INFO] 所有 Python 依赖已就绪，跳过安装。")


def _seed_files(backend: OpenSandboxBackend) -> None:
    """
    将本地技能文件上传到沙箱。

    AGENTS.md 已迁移到 StoreBackend（全局共享，不经过沙箱）。
    仅上传在沙箱中尚不存在的文件，避免覆盖已更新的内容。
    """
    file_mapping: List[Tuple[Path, str]] = []

    # 遍历 skills 目录，添加所有技能文件
    skills_base = Path(LOCAL_SKILLS_DIR)
    if skills_base.exists():
        for skill_dir in skills_base.iterdir():
            if not skill_dir.is_dir():
                continue
            for local_file in skill_dir.rglob("*"):
                if local_file.is_file():
                    rel = local_file.relative_to(skills_base).as_posix()
                    sandbox_path = f"{SANDBOX_SKILLS_ROOT}/{rel}"
                    file_mapping.append((local_file, sandbox_path))

    # 收集需要上传的文件
    to_upload: List[Tuple[str, bytes]] = []
    for local_path, sandbox_path in file_mapping:
        if not local_path.exists():
            continue
        local_content = local_path.read_bytes()
        # 用 test -f 检测文件是否存在（无 ERROR 日志），避免 download_files 对 404 打 ERROR
        check = backend.execute(f"test -f {sandbox_path}")
        if check.exit_code == 0:
            try:
                results = backend.download_files([sandbox_path])
                if results and results[0].content and not results[0].error:
                    remote_content = results[0].content
                    if isinstance(remote_content, str):
                        remote_content = remote_content.encode("utf-8")
                    if remote_content == local_content:
                        continue
            except Exception:
                pass
        to_upload.append((sandbox_path, local_content))

    if to_upload:
        print(f"[INFO] 正在上传 {len(to_upload)} 个基础文件...")
        backend.upload_files(to_upload)
        print("[INFO] 基础文件上传完成。")
    else:
        print("[INFO] 所有基础文件已就绪，无需上传。")