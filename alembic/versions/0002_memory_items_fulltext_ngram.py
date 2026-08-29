"""memory_items FULLTEXT ngram 索引

Revision ID: 0002
Revises: dccb397a0b8d
Create Date: 2026-08-08

方案 17.2：FULLTEXT/ngram 索引由单独 migration 创建；
若目标 MySQL 未安装/启用 ngram parser，migration 必须明确失败，不能静默退化。
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "dccb397a0b8d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ngram parser 是 MySQL 内置解析器；若目标实例不可用，此语句直接失败，
    # migration 失败即 fail closed（方案 17.2）。
    op.execute(
        "CREATE FULLTEXT INDEX ft_memory_items_content "
        "ON memory_items (content) WITH PARSER ngram"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX ft_memory_items_content ON memory_items")
