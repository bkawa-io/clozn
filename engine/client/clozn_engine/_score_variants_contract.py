"""Generated from protocol/fixtures/score_variants.json. Do not edit by hand."""

SCORE_VARIANTS_ENDPOINT = '/v1/score-variants'
SCORE_VARIANTS_CAPABILITY = 'score_variants'
SCORE_VARIANTS_SCHEMA = 'clozn.score-variants-worker.v1'
SCORE_VARIANTS_MAX_VARIANTS = 64
SCORE_VARIANTS_MAX_VARIANT_ID_BYTES = 128
SCORE_VARIANTS_MAX_PROMPT_ROWS_PER_CONTEXT = 16
SCORE_VARIANTS_MAX_CHECKPOINT_ID_BYTES = 512
SCORE_VARIANTS_MAX_WORKER_GENERATION_ID_BYTES = 128
SCORE_VARIANTS_REQUEST_KEYS = ('variants', 'continuation_ids', 'generation_state',)
SCORE_VARIANT_KEYS = ('id', 'prompt_ids',)
SCORE_VARIANTS_GENERATION_STATE_KEYS = ('checkpoint_id', 'worker_generation_id', 'baseline_variant_id',)
SCORE_VARIANTS_TOP_STATUSES = ('completed', 'partial', 'failed', 'cancelled',)
SCORE_VARIANTS_ROW_STATUSES = ('completed', 'failed', 'cancelled', 'not_started', 'unsupported',)
SCORE_VARIANTS_STRATEGY_PREFIX = 'native_'
SCORE_VARIANTS_NUMERICAL_REGIME = 'screening_unqualified'
SCORE_VARIANTS_EVIDENCE_USE = 'screening'
SCORE_VARIANTS_QUALIFIED = False
