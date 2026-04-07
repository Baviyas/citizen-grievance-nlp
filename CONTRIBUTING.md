# Contributing Guide

Thank you for contributing to the Citizen Grievance NLP project. This guide explains how our team of three works together and how you can make contributions in a consistent, efficient way.

## Getting Started

1. Fork the repository or work directly in the shared repository if your team agrees.
2. Create a branch for each feature or fix:
   - `feature/<short-description>`
   - `fix/<short-description>`
   - `docs/<short-description>`
3. Use descriptive commit messages and keep changes scoped.

## Development Workflow

- Keep your branch up to date with `main` or the agreed base branch.
- Open a pull request when your change is ready.
- Assign at least one teammate to review your PR.
- Address review feedback before merging.
- Use `main` only for approved, merged work.

## Testing

- Run any available tests before submitting a PR.
- Verify the app starts successfully after your changes.
- Ensure new code does not break existing behavior.

## Commit Message Template

Use a simple format like:

```
<type>: <short summary>

<detailed description>
```

Examples:
- `feat: add Streamlit redirect support`
- `fix: correct API client timeout handling`
- `docs: update frontend setup instructions`
