#!/usr/bin/env bash
# Create Nebius Object Storage (S3-compatible) credentials for the *currently
# logged-in user* (no service account needed) and configure an AWS CLI profile.
#
# Usage:
#   ./setup-s3-creds.sh [options]
#
# Options (all optional; env var equivalents in brackets):
#   -P, --aws-profile NAME   AWS CLI profile to write        [AWS_PROFILE_NAME] (default: nebius)
#   -n, --nebius-profile N   Nebius CLI profile to use       [NEBIUS_PROFILE]   (default: CLI default)
#   -p, --parent-id ID       Project/tenant ID for the key   [PARENT_ID]        (default: from nebius config)
#   -u, --user-id ID         Account ID to own the key       [USER_ID]          (default: from whoami)
#   -r, --region REGION      Object Storage region           [NEBIUS_REGION]    (default: eu-north1)
#   -d, --days N             Key lifetime in days            [KEY_DAYS]         (default: 30)
#   -e, --print-env          Also print `export AWS_...=` lines (incl. NEBIUS_IAM_TOKEN) to stdout
#   -h, --help
#
# Examples:
#   ./setup-s3-creds.sh                         # configure AWS profile "nebius"
#   aws --profile nebius s3 ls
#
#   eval "$(./setup-s3-creds.sh --print-env)"   # also export creds into current shell
set -euo pipefail

AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-nebius}"
NEBIUS_PROFILE="${NEBIUS_PROFILE:-}"
PARENT_ID="${PARENT_ID:-}"
USER_ID="${USER_ID:-}"
NEBIUS_REGION="${NEBIUS_REGION:-eu-north1}"
KEY_DAYS="${KEY_DAYS:-30}"
PRINT_ENV=0

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }
log() { echo "[nebius-s3] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -P|--aws-profile)    AWS_PROFILE_NAME="$2"; shift 2 ;;
    -n|--nebius-profile) NEBIUS_PROFILE="$2"; shift 2 ;;
    -p|--parent-id)      PARENT_ID="$2"; shift 2 ;;
    -u|--user-id)        USER_ID="$2"; shift 2 ;;
    -r|--region)         NEBIUS_REGION="$2"; shift 2 ;;
    -d|--days)           KEY_DAYS="$2"; shift 2 ;;
    -e|--print-env)      PRINT_ENV=1; shift ;;
    -h|--help)           usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

for bin in nebius jq aws; do
  command -v "$bin" >/dev/null || die "'$bin' not found in PATH"
done

nb() {
  if [[ -n "$NEBIUS_PROFILE" ]]; then nebius --profile "$NEBIUS_PROFILE" "$@"; else nebius "$@"; fi
}

# --- Resolve IDs -------------------------------------------------------------
log "Checking Nebius login..."
PROFILE_JSON="$(nb iam whoami --format json)" || die "not logged in; run 'nebius iam get-access-token' to authenticate"

if [[ -z "$PARENT_ID" ]]; then
  PARENT_ID="$(nb config get parent-id 2>/dev/null || true)"
  if [[ -z "$PARENT_ID" ]]; then
    PARENT_ID="$(nb config list 2>/dev/null | awk -F': *' '$1=="parent-id"{print $2}')"
  fi
  [[ -n "$PARENT_ID" ]] || die "could not determine parent-id; pass --parent-id"
fi

# Candidate account IDs: explicit, else the user account then active tenant user accounts.
if [[ -n "$USER_ID" ]]; then
  CANDIDATES=("$USER_ID")
else
  CANDIDATES=()
  while IFS= read -r id; do [[ -n "$id" ]] && CANDIDATES+=("$id"); done < <(
    jq -r '.user_profile.id,
           (.user_profile.tenants[]? | select(.tenant_user_account_state=="ACTIVE") | .tenant_user_account_id)' \
      <<<"$PROFILE_JSON"
  )
  [[ ${#CANDIDATES[@]} -gt 0 ]] || die "could not determine user id; pass --user-id"
fi

# --- Expiry ------------------------------------------------------------------
if date -u -v+1d >/dev/null 2>&1; then
  EXPIRES_AT="$(date -u -v+"${KEY_DAYS}"d +%Y-%m-%dT%H:%M:%SZ)"        # BSD/macOS
else
  EXPIRES_AT="$(date -u -d "+${KEY_DAYS} days" +%Y-%m-%dT%H:%M:%SZ)"   # GNU
fi

# --- Create access key -------------------------------------------------------
KEY_NAME="aws-cli-$(whoami)-$(date +%Y%m%d%H%M%S)"
ACCESS_KEY_ID=""
for acct in "${CANDIDATES[@]}"; do
  log "Creating access key '$KEY_NAME' for $acct in $PARENT_ID (expires $EXPIRES_AT)..."
  if out="$(nb iam v2 access-key create \
        --parent-id "$PARENT_ID" \
        --account-user-account-id "$acct" \
        --name "$KEY_NAME" \
        --description "AWS CLI key created by nebius-utils/setup-s3-creds.sh" \
        --expires-at "$EXPIRES_AT" \
        --secret-delivery-mode explicit \
        --format json 2>&1)"; then
    ACCESS_KEY_ID="$(jq -r '.metadata.id // .resource_id // empty' <<<"$out" 2>/dev/null || true)"
    [[ -n "$ACCESS_KEY_ID" ]] && break
    log "Unexpected create output: $out"
  else
    log "Create failed for $acct: $(tail -n 3 <<<"$out")"
  fi
done
[[ -n "$ACCESS_KEY_ID" ]] || die "could not create an access key (check permissions in $PARENT_ID)"
log "Created access key $ACCESS_KEY_ID"

# --- Fetch AWS key id + secret -------------------------------------------------
KEY_JSON="$(nb iam v2 access-key get --id "$ACCESS_KEY_ID" --format json)"
AWS_KEY_ID="$(jq -r '.status.aws_access_key_id // empty' <<<"$KEY_JSON")"
AWS_SECRET="$(jq -r '.status.secret // empty' <<<"$KEY_JSON")"
if [[ -z "$AWS_SECRET" ]]; then
  SECRET_JSON="$(nb iam v2 access-key get-secret --id "$ACCESS_KEY_ID" --format json)"
  AWS_SECRET="$(jq -r '[.. | objects | .secret? | strings] | first // empty' <<<"$SECRET_JSON")"
fi
[[ -n "$AWS_KEY_ID" && -n "$AWS_SECRET" ]] || die "failed to read key id/secret for $ACCESS_KEY_ID"

# --- Configure AWS CLI profile ------------------------------------------------
ENDPOINT="https://storage.${NEBIUS_REGION}.nebius.cloud"
aws configure set aws_access_key_id     "$AWS_KEY_ID" --profile "$AWS_PROFILE_NAME"
aws configure set aws_secret_access_key "$AWS_SECRET" --profile "$AWS_PROFILE_NAME"
aws configure set region                "$NEBIUS_REGION" --profile "$AWS_PROFILE_NAME"
aws configure set endpoint_url          "$ENDPOINT" --profile "$AWS_PROFILE_NAME"
log "Configured AWS profile '$AWS_PROFILE_NAME' -> $ENDPOINT"
log "Try: aws --profile $AWS_PROFILE_NAME s3 ls"
log "Revoke later with: nebius iam v2 access-key delete --id $ACCESS_KEY_ID"

if [[ $PRINT_ENV -eq 1 ]]; then
  TOKEN="$(nb iam get-access-token)"
  cat <<EOF
export AWS_ACCESS_KEY_ID='$AWS_KEY_ID'
export AWS_SECRET_ACCESS_KEY='$AWS_SECRET'
export AWS_REGION='$NEBIUS_REGION'
export AWS_DEFAULT_REGION='$NEBIUS_REGION'
export AWS_ENDPOINT_URL='$ENDPOINT'
export NEBIUS_IAM_TOKEN='$TOKEN'
EOF
fi
