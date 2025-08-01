"""Add reset password

Revision ID: d73b566e8ba1
Revises: 0909e9269c07
Create Date: 2025-07-31 13:49:53.076551

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd73b566e8ba1'
down_revision: Union[str, Sequence[str], None] = '0909e9269c07'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('email_verification_codes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('code', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_email_verification_codes_id'), 'email_verification_codes', ['id'], unique=False)
    op.create_table('password_reset_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('token', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('token')
    )
    op.create_index(op.f('ix_password_reset_tokens_id'), 'password_reset_tokens', ['id'], unique=False)
    op.add_column('users', sa.Column('is_verified', sa.Boolean(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('users', 'is_verified')
    op.drop_index(op.f('ix_password_reset_tokens_id'), table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')
    op.drop_index(op.f('ix_email_verification_codes_id'), table_name='email_verification_codes')
    op.drop_table('email_verification_codes')
    # ### end Alembic commands ###
