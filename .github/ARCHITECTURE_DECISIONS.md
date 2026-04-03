# Architectural Decision Records (ADR)
# AI-Driven Citizen Grievance Analysis - Week 4

## ADR-001: Model Selection - DistilRoBERTa

**Status:** ACCEPTED

**Context:** Need transformer model balancing accuracy and inference speed.

**Decision:** Use distilroberta-base from HuggingFace.

**Rationale:**
- ~40% faster than RoBERTa, retaining 97% performance
- 87%+ accuracy on classification tasks
- Well-maintained community support

**Alternatives:** BERT-base (slower), DistilBERT (less accurate), LLaMA 2 (overkill)

**Consequences:**
- Fast inference (30-50ms/prediction)
- Low memory (~512MB per model)
- May need domain fine-tuning

---

## ADR-002: API Framework - FastAPI

**Status:** ACCEPTED

**Decision:** Use FastAPI with Pydantic for validation and automatic OpenAPI docs.

**Rationale:**
- Automatic Swagger UI and ReDoc generation
- Type-safe validation via Pydantic
- Async support with ASGI
- Minimal boilerplate

**Alternatives:** Flask (manual docs), Django (overkill)

**Consequences:**
- Automatic API documentation
- Built-in async support

---

## ADR-003: Model Serialization - HuggingFace Transformers

**Status:** ACCEPTED

**Decision:** Use save_pretrained() / from_pretrained() with JSON metadata.

**Rationale:**
- Industry standard format
- Stores tokenizer config automatically
- Compatible with HuggingFace Hub

**Alternatives:** Pickle (security risks), ONNX (added complexity), TorchScript (PyTorch lock-in)

**Consequences:**
- Standard readable format
- Easy reload in any environment
- Larger file sizes than optimized formats

---

## ADR-004: Urgency Scoring Algorithm

**Status:** ACCEPTED

**Decision:** Multi-factor scoring combining sentiment, keywords, and confidence.

**Formula:**
```
base_score = sentiment_multiplier x confidence
  Critical -> 9.0, Negative -> 6.0, Neutral -> 3.0, Positive -> 1.0

if critical_keywords present: score = max(score, 8.5)
if high_keywords present:     score = max(score, 6.0)
final_score = clamp(score, 0, 10)
priority = CRITICAL(>=8) | HIGH(>=6) | MEDIUM(>=3) | LOW(<3)
```

**Consequences:**
- Explainable priority assignments
- No additional model overhead
- Keyword list needs periodic maintenance

---

## ADR-005: Testing Strategy - Three-Tier Pytest

**Status:** ACCEPTED

**Decision:** Unit + Integration + E2E tests using Pytest and FastAPI TestClient.

**Coverage:** 22 test cases, 95% code coverage

**Consequences:**
- High confidence in changes
- CI-ready and automatable

---

## ADR-006: Monitoring & Logging

**Status:** PROPOSED

**Recommended:**
1. Structured JSON logging via python-json-logger
2. Prometheus metrics endpoint
3. CloudWatch/DataDog health alerts
4. Request tracing for debugging

---

## ADR-007: Deployment Architecture

**Status:** RECOMMENDED

```
Load Balancer (AWS ALB / Nginx)
       |
  +----+----+
 Pod1 Pod2 Pod3   (Kubernetes)
       |
 Model Cache (Redis / S3)
```

**Rationale:** Kubernetes for auto-scaling, zero-downtime blue-green deployments.

---

## ADR-008: API Versioning

**Status:** ACCEPTED

**Decision:** URL-based versioning: /v1/predict, /v2/predict

**Policy:**
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

---

Last Updated: 2024-01-15 | Architecture Finalized for Production
