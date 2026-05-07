"""Utilitaires de sécurité — authentification HTTPBasic avec secrets.compare_digest."""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    """Valide les identifiants Basic auth avec une comparaison à temps constant."""
    api_user = os.environ.get("API_USERNAME", "admin")
    api_pass = os.environ.get("API_PASSWORD", "changeme")
    valid_user = secrets.compare_digest(credentials.username, api_user)
    valid_pass = secrets.compare_digest(credentials.password, api_pass)
    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
