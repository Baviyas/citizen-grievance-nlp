
## 🏗 System Overview

The system follows a **decoupled client–server architecture**. A **FastAPI** backend manages machine learning inference and business logic, while a **Streamlit** frontend handles the user interface. This separation ensures scalability, maintainability, and a lightweight UI layer.


## 📂 Project Structure

```text

├── api/                # FastAPI backend (routes, logic, tests)
├── frontend/           # Streamlit UI (pages, utilities)
├── models/             # Serialized ML artifacts
└── docs/               # Architecture and design documentation
```

## 🛠 Tech Stack

| Layer         | Technology                | Purpose                |
| ------------- | ------------------------- | ---------------------- |
| Frontend      | Streamlit                 | UI & visualization     |
| Backend       | FastAPI                   | API & async processing |
| Validation    | Pydantic                  | Schema validation      |
| Inference     | Scikit-learn / TensorFlow | Model execution        |
| Communication | REST / JSON               | Data exchange          |

---

## 🔄 Data Flow

1. User interacts with the Streamlit UI
2. Frontend sends a JSON request to FastAPI
3. Backend validates input via Pydantic
4. Model inference runs using stored artifacts
5. Response is returned and rendered in the UI

> Swagger docs available at: `http://localhost:8000/docs`

## 📏 Design Principles

* **Loose Coupling:** Independent frontend and backend deployment
* **Separation of Concerns:** UI, logic, and models are isolated
* **Stateless API:** Each request is processed independently
* **Artifact Isolation:** Models remain external to application logic
* **Robust Errors:** Standard HTTP status codes enforced

## 🚀 Local Setup

```bash
# Backend
cd api
python app.py

# Frontend
cd frontend
streamlit run app.py --server.port 8501
```
