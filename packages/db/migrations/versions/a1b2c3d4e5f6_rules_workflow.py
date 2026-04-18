"""rules workflow — rules table, poems.qafiya_rule_id, drop classifier_guidelines

Revision ID: a1b2c3d4e5f6
Revises: fbdf0df5cbfe
Create Date: 2026-04-17

Introduces the rule-based qafiya workflow:
- `rules` table: user-proposed rules, status tracks lifecycle (proposed → coded → applied)
- `poems.qafiya_rule_id` FK: which rule populated a poem's qafiya fields
- drops `classifier_guidelines` (superseded by `rules`)
- adds `qafiya_annotations.correct_wasl` (bakes prior hand-applied ALTER)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'fbdf0df5cbfe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'rules',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('code', sa.String(length=16), nullable=True),
        sa.Column('title_ar', sa.Text(), nullable=False),
        sa.Column('description_ar', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='proposed'),
        sa.Column('function_name', sa.String(length=128), nullable=True),
        sa.Column('poems_matched', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('example_poem_id', sa.UUID(), nullable=True),
        sa.Column('example_qafiya_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('coded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['example_poem_id'], ['poems.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
    )
    op.create_index(op.f('ix_rules_status'), 'rules', ['status'], unique=False)

    op.add_column(
        'poems',
        sa.Column('qafiya_rule_id', sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        'fk_poems_qafiya_rule_id', 'poems', 'rules',
        ['qafiya_rule_id'], ['id'], ondelete='SET NULL',
    )
    op.create_index(op.f('ix_poems_qafiya_rule_id'), 'poems', ['qafiya_rule_id'], unique=False)

    op.add_column(
        'qafiya_annotations',
        sa.Column('correct_wasl', sa.String(length=4), nullable=True),
    )

    op.drop_table('classifier_guidelines')


def downgrade() -> None:
    op.create_table(
        'classifier_guidelines',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('guideline', sa.Text(), nullable=False),
        sa.Column('example_poem_id', sa.UUID(), nullable=True),
        sa.Column('applied', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['example_poem_id'], ['poems.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.drop_column('qafiya_annotations', 'correct_wasl')

    op.drop_index(op.f('ix_poems_qafiya_rule_id'), table_name='poems')
    op.drop_constraint('fk_poems_qafiya_rule_id', 'poems', type_='foreignkey')
    op.drop_column('poems', 'qafiya_rule_id')

    op.drop_index(op.f('ix_rules_status'), table_name='rules')
    op.drop_table('rules')
