"""Azure Key Vault bootstrap — the pod's API keys and shared secrets live in
kv-ntm-apps, not in the host's environment.

Call ``load()`` once, before anything reads ``os.environ``: each secret found in
the vault is written into the process environment under the env var name the
rest of the code already uses, so ``cw_client``, ``hudu_client`` etc. stay
env-driven and need no vault awareness. Vault values win over anything the host
set, so a stale variable left behind on the platform can never shadow the vault.

Configuration (all read from the environment):

  AZURE_KEY_VAULT_URL   https://kv-ntm-apps.vault.azure.net — unset disables the
                        vault entirely (local dev / tests fall back to .env).
  AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
                        the "hercules-tech-keyvault-reader" service principal,
                        which holds Key Vault Secrets User on the vault. Picked up
                        by DefaultAzureCredential, which also honours a managed
                        identity (Azure Container Apps) or an ``az login`` session
                        (a developer machine) when those three are absent.

A vault that is configured but unreachable, or a credential that is refused, is
fatal: the pod refuses to start rather than run with no ConnectWise keys. A
single secret missing from the vault is logged and skipped so a key that has
not been uploaded yet can still be supplied through the environment.
"""

import logging
import os

VAULT_URL_VAR = "AZURE_KEY_VAULT_URL"

# env var the code reads -> secret name in kv-ntm-apps.
# Shared integrations use the plain name (one ConnectWise key set for every
# service); secrets that only exist between the pod and the Hercules server
# carry the "hercules-" prefix.
SECRETS: dict[str, str] = {
    "CW_CLIENT_ID": "cw-client-id",
    "CW_PUBLIC_KEY": "cw-public-key",
    "CW_PRIVATE_KEY": "cw-private-key",
    "HUDU_API_KEY": "hudu-api-key",
    "OPENROUTER_API_KEY": "openrouter-api-key",
    "SCREENCONNECT_API_SECRET": "screenconnect-api-secret",
    "SMS_AGENT_SECRET": "sms-agent-tech-api-secret",
    "PIA_API_SECRET": "pia-api-secret",
    "POD_SECRET": "hercules-pod-secret",
    "LIVE_BRIDGE_SECRET": "hercules-live-bridge-secret",
    "CALL_BRIDGE_SECRET": "hercules-ingest-secret",
}


def is_configured() -> bool:
    return bool(os.getenv(VAULT_URL_VAR, "").strip())


def load() -> list[str]:
    """Pull every secret in SECRETS into os.environ. Returns the env var names
    that were set from the vault (empty when the vault is not configured)."""
    vault_url = os.getenv(VAULT_URL_VAR, "").strip()
    if not vault_url:
        print(f"[vault] {VAULT_URL_VAR} not set — secrets come from the environment / .env")
        return []

    # The SDK's credential chain and HTTP pipeline are chatty at INFO.
    logging.getLogger("azure").setLevel(logging.WARNING)

    from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    client = SecretClient(vault_url=vault_url, credential=credential)

    loaded: list[str] = []
    missing: list[str] = []
    for env_name, secret_name in SECRETS.items():
        try:
            value = client.get_secret(secret_name).value
        except ResourceNotFoundError:
            missing.append(secret_name)
            continue
        except ClientAuthenticationError as exc:
            raise RuntimeError(
                f"[vault] authentication to {vault_url} failed — check AZURE_TENANT_ID / "
                f"AZURE_CLIENT_ID / AZURE_CLIENT_SECRET: {exc}"
            ) from exc
        except Exception as exc:  # network, 403 (role not assigned), throttling
            raise RuntimeError(f"[vault] could not read '{secret_name}' from {vault_url}: {exc}") from exc
        if value:
            os.environ[env_name] = value
            loaded.append(env_name)

    print(f"[vault] loaded {len(loaded)} secret(s) from {vault_url}: {', '.join(loaded) or '-'}")
    if missing:
        print(f"[vault] not in vault (environment fallback): {', '.join(missing)}")
    return loaded
