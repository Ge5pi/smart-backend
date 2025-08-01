"""empty message

Revision ID: 9ca51222d545
Revises: 
Create Date: 2025-07-23 16:58:12.137768

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9ca51222d545'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('an_files', sa.Column('file_uid', sa.String(), nullable=False))
    op.add_column('an_files', sa.Column('datetime_created', sa.DateTime(), nullable=False))
    op.add_column('an_files', sa.Column('file_name', sa.String(), nullable=False))
    op.add_column('an_files', sa.Column('s3_path', sa.String(), nullable=False))
    op.alter_column('an_files', 'user_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    op.drop_index(op.f('ix_an_files_filename'), table_name='an_files')
    op.create_index(op.f('ix_an_files_file_uid'), 'an_files', ['file_uid'], unique=True)
    op.create_index(op.f('ix_an_files_user_id'), 'an_files', ['user_id'], unique=False)
    op.create_unique_constraint(None, 'an_files', ['s3_path'])
    op.drop_column('an_files', 'created_at')
    op.drop_column('an_files', 'filename')
    op.alter_column('users', 'email',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.alter_column('users', 'hashed_password',
               existing_type=sa.VARCHAR(),
               nullable=False)
    op.drop_column('users', 'created_at')
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True))
    op.alter_column('users', 'hashed_password',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.alter_column('users', 'email',
               existing_type=sa.VARCHAR(),
               nullable=True)
    op.add_column('an_files', sa.Column('filename', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('an_files', sa.Column('created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True))
    op.drop_constraint(None, 'an_files', type_='unique')
    op.drop_index(op.f('ix_an_files_user_id'), table_name='an_files')
    op.drop_index(op.f('ix_an_files_file_uid'), table_name='an_files')
    op.create_index(op.f('ix_an_files_filename'), 'an_files', ['filename'], unique=False)
    op.alter_column('an_files', 'user_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.drop_column('an_files', 's3_path')
    op.drop_column('an_files', 'file_name')
    op.drop_column('an_files', 'datetime_created')
    op.drop_column('an_files', 'file_uid')
    # ### end Alembic commands ###
