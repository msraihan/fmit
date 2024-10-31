# This package describes how to authenticate users in a Streamlit app running in Databricks.

# The get_user_info function retrieves the user's information from the WebSocket headers.
# The get_user_credentials_provider function returns the user's credentials provider, which is used to authenticate the user to the Databricks workspace.
# The get_service_principal_credentials_provider function returns the app's service principal credentials provider, which is used to authenticate the app to the Databricks workspace.

import os
import io
from typing import Dict, Optional
from streamlit.web.server.websocket_headers import _get_websocket_headers
from databricks.sdk.core import Config, oauth_service_principal, CredentialsProvider
# from databricks.sdk.core import Config, HeaderFactory, oauth_service_principal, CredentialsProvider

from databricks.sdk import WorkspaceClient
from databricks import sql

from databricks import sdk

import streamlit as st
def get_user_info():
    headers = st.context.headers
    
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        access_token=headers.get("X-Forwarded-Access-Token")
    )

def get_app_service_principal_info():
    return dict(
        client_id=os.getenv('DATABRICKS_CLIENT_ID'),
        client_secret=os.getenv('DATABRICKS_CLIENT_SECRET'),
    )

def get_user_credentials_provider() -> CredentialsProvider:
    """Returns a credentials provider for the current user.
    This is so the same method can be used in Databricks SDK and SQL connector.
    It looks complicated, but it is just using the access_token from the user_info."""

    def inner(cfg: Optional[Config] = None) -> Dict[str, str]:
        user_info = get_user_info()
        if not user_info.get("access_token"):
            raise ValueError("User access token not found. Please make sure the feature is enabled in your workspace.")
        static_credentials = {'Authorization': f'Bearer {user_info.get("access_token")}'}
        return lambda: static_credentials
    
    inner.auth_type = lambda: 'app-user-oauth'

    return inner
    
def get_app_credentials_provider() -> CredentialsProvider:
    """Returns a credentials provider for the app service principal.
    This is so the same method can be used in Databricks SDK and SQL connector.
    It looks complicated, but it is just using the oauth_service_principal method."""

    def inner(cfg: Optional[Config] = None) -> Dict[str, str]:
        if cfg is None:
              cfg = Config()
        return oauth_service_principal(cfg)

    inner.auth_type = lambda: 'app-service-principal-oauth'

    return inner

def get_user_workspace_client() -> WorkspaceClient:
    """Returns a WorkspaceClient for the current user."""
    return WorkspaceClient(config=Config(credentials_provider=get_user_credentials_provider()))


def get_app_workspace_client() -> WorkspaceClient:
    """Returns a WorkspaceClient for the app service principal. Uses OAuth for authentication."""
    return WorkspaceClient(config=Config(credentials_provider=get_app_credentials_provider()))


def get_user_sql_connection(http_path):
    """Returns a SQL connection for the current user. 
    This is meant to be used for short-lived connections. For long-lived connections, use get_app_sql_connection."""
    return sql.connect(
        server_hostname=os.getenv('DATABRICKS_HOST'),
        http_path=http_path,
        credentials_provider=get_user_credentials_provider()
    )

def get_app_sql_connection(http_path):
    """Returns a SQL connection for the app service principal."""
    return sql.connect(
        server_hostname=os.getenv('DATABRICKS_HOST'),
        http_path=http_path,
        credentials_provider=get_app_credentials_provider()
    )
