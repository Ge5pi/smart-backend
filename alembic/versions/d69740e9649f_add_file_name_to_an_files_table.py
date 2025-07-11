"""Add file_name to an_files table

Revision ID: d69740e9649f
Revises: 53b89a98122c
Create Date: 2025-07-08 15:59:20.371932

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd69740e9649f'
down_revision: Union[str, Sequence[str], None] = '53b89a98122c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('an_files', sa.Column('file_name', sa.String(), nullable=False))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('an_files', 'file_name')
    # ### end Alembic commands ###
