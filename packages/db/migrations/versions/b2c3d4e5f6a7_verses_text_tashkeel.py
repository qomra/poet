"""verses.text_tashkeel column for model-inferred diacritization

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-18

Adds `verses.text_tashkeel` — populated by an ETL step that calls the
Fine-Tashkeel service. Distinct from `text_diacritized`, which is reserved
for diacritics present in the source data (currently always NULL for ashaar).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('verses', sa.Column('text_tashkeel', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('verses', 'text_tashkeel')
