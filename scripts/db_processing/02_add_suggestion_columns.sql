-- SQL Commands to add columns for LLM suggestions and review status
-- These should be run against your Supabase "Knowledge Database"

-- Add columns to the 'law' table
ALTER TABLE public.law
ADD COLUMN IF NOT EXISTS suggested_classification TEXT[],
ADD COLUMN IF NOT EXISTS suggested_sharia_influence BOOLEAN,
ADD COLUMN IF NOT EXISTS classification_confidence FLOAT,
ADD COLUMN IF NOT EXISTS classification_reasoning TEXT,
ADD COLUMN IF NOT EXISTS classification_llm_model TEXT, -- To store which LLM version generated it
ADD COLUMN IF NOT EXISTS classification_processed_at TIMESTAMPTZ, -- Timestamp of LLM processing
ADD COLUMN IF NOT EXISTS classification_needs_review BOOLEAN DEFAULT TRUE;

COMMENT ON COLUMN public.law.suggested_classification IS 'LLM-suggested categories for the law.';
COMMENT ON COLUMN public.law.suggested_sharia_influence IS 'LLM-suggested Sharia influence assessment.';
COMMENT ON COLUMN public.law.classification_confidence IS 'LLM''s confidence score for the law classification.';
COMMENT ON COLUMN public.law.classification_reasoning IS 'LLM''s reasoning for the law classification.';
COMMENT ON COLUMN public.law.classification_llm_model IS 'Identifier for the LLM model and version used for classification.';
COMMENT ON COLUMN public.law.classification_processed_at IS 'Timestamp when the LLM classification was processed.';
COMMENT ON COLUMN public.law.classification_needs_review IS 'Flag indicating if the LLM classification suggestions need human review (TRUE = needs review, FALSE = reviewed/approved).';

-- Add columns to the 'law_articles' table
ALTER TABLE public.law_articles
ADD COLUMN IF NOT EXISTS suggested_tags TEXT[],
ADD COLUMN IF NOT EXISTS tagging_confidence FLOAT,
ADD COLUMN IF NOT EXISTS tagging_reasoning TEXT,
ADD COLUMN IF NOT EXISTS tagging_llm_model TEXT, -- To store which LLM version generated it
ADD COLUMN IF NOT EXISTS tagging_processed_at TIMESTAMPTZ, -- Timestamp of LLM processing
ADD COLUMN IF NOT EXISTS tags_need_review BOOLEAN DEFAULT TRUE;

COMMENT ON COLUMN public.law_articles.suggested_tags IS 'LLM-suggested topical tags for the article.';
COMMENT ON COLUMN public.law_articles.tagging_confidence IS 'LLM''s confidence score for the article tagging.';
COMMENT ON COLUMN public.law_articles.tagging_reasoning IS 'LLM''s reasoning for the article tagging.';
COMMENT ON COLUMN public.law_articles.tagging_llm_model IS 'Identifier for the LLM model and version used for tagging.';
COMMENT ON COLUMN public.law_articles.tagging_processed_at IS 'Timestamp when the LLM tagging was processed.';
COMMENT ON COLUMN public.law_articles.tags_need_review IS 'Flag indicating if the LLM tag suggestions need human review (TRUE = needs review, FALSE = reviewed/approved).';

-- Optional: Create indexes on the '_needs_review' columns if you plan to query them frequently
-- CREATE INDEX IF NOT EXISTS idx_law_classification_needs_review ON public.law (classification_needs_review) WHERE classification_needs_review = TRUE;
-- CREATE INDEX IF NOT EXISTS idx_law_articles_tags_need_review ON public.law_articles (tags_need_review) WHERE tags_need_review = TRUE;

SELECT 'Schema modification script complete. Review and apply to your database.';
