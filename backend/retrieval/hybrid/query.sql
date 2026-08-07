WITH dense AS (
    SELECT chunk_id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
    FROM chunk_embeddings
    WHERE document_id = $2
    ORDER BY embedding <=> $1::vector
    LIMIT 40
),
lexical AS (
    SELECT chunk_id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(lexeme, plainto_tsquery('english', $3)) DESC
           ) AS rank
    FROM chunk_embeddings
    WHERE document_id = $2
      AND lexeme @@ plainto_tsquery('english', $3)
    LIMIT 40
)
SELECT c.chunk_id,
       c.chunk_text,
       p.parent_text,
       COALESCE(1.0 / (60 + d.rank), 0) + COALESCE(1.0 / (60 + l.rank), 0) AS fused
FROM chunk_embeddings c
JOIN parent_chunks p ON p.parent_id = c.parent_id
LEFT JOIN dense d ON d.chunk_id = c.chunk_id
LEFT JOIN lexical l ON l.chunk_id = c.chunk_id
WHERE d.chunk_id IS NOT NULL OR l.chunk_id IS NOT NULL
ORDER BY fused DESC
LIMIT 20;