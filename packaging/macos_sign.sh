#!/bin/bash
# Sign (and, when Apple credentials are present, notarize) the macOS build.
#
#   packaging/macos_sign.sh dist/PatchClampAnalysis.app PatchClampAnalysis-macOS-arm64
#
# Produces <name>.zip (in-app full update) and <name>.dmg (first install).
#
# Without secrets: the app gets a complete ad-hoc signature ("-"), sealed and
# verified, so macOS remembers the user's one-time "Open Anyway" approval.
#
# With a paid Apple Developer account (optional, dormant until the secrets exist):
#   MACOS_CERT_P12_BASE64   Developer ID Application certificate (.p12), base64
#   MACOS_CERT_PASSWORD     password of that .p12
#   and either an App Store Connect API key
#     APPLE_API_KEY_P8_BASE64, APPLE_API_KEY_ID, APPLE_API_ISSUER_ID
#   or an Apple ID
#     APPLE_ID, APPLE_APP_PASSWORD (app-specific password), APPLE_TEAM_ID
# the app is signed with hardened runtime, notarized and stapled; Gatekeeper
# then opens it without any prompt.
set -euo pipefail

APP="$1"
NAME="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
ENT="$HERE/macos_entitlements.plist"

IDENTITY="-"
NOTARIZE=false
if [ -n "${MACOS_CERT_P12_BASE64:-}" ]; then
  KC="$RUNNER_TEMP/signing.keychain-db"
  KCPW="$(uuidgen)"
  echo "$MACOS_CERT_P12_BASE64" | base64 --decode > "$RUNNER_TEMP/cert.p12"
  security create-keychain -p "$KCPW" "$KC"
  security set-keychain-settings -lut 3600 "$KC"
  security unlock-keychain -p "$KCPW" "$KC"
  security import "$RUNNER_TEMP/cert.p12" -k "$KC" -P "${MACOS_CERT_PASSWORD:-}" -T /usr/bin/codesign
  security set-key-partition-list -S apple-tool:,apple: -s -k "$KCPW" "$KC" >/dev/null
  security list-keychains -d user -s "$KC" $(security list-keychains -d user | tr -d '"')
  rm -f "$RUNNER_TEMP/cert.p12"
  IDENTITY="$(security find-identity -v -p codesigning "$KC" | grep 'Developer ID Application' | head -1 | sed -E 's/.*"(.*)"/\1/')"
  test -n "$IDENTITY" || { echo "no 'Developer ID Application' identity in the .p12"; exit 1; }
  echo "signing with: $IDENTITY"
  if [ -n "${APPLE_API_KEY_P8_BASE64:-}" ] || [ -n "${APPLE_ID:-}" ]; then NOTARIZE=true; fi
else
  echo "::notice::No Apple certificate configured: ad-hoc signing (users approve once on first open)."
fi

sign() {   # sign one item; hardened runtime + timestamp only for a real identity
  if [ "$IDENTITY" = "-" ]; then
    codesign --force --sign - "$@"
  else
    codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$IDENTITY" "$@"
  fi
}

# Inside out: every Mach-O file first, then the bundle itself, which seals all resources.
find "$APP/Contents" -type f -print0 | while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q 'Mach-O'; then sign "$f"; fi
done
# nested bundles (Qt *.framework), deepest first, so each one's resources are sealed
find "$APP/Contents" -type d \( -name '*.framework' -o -name '*.app' \) \
  | awk '{ print gsub("/", "/"), $0 }' | sort -rn | cut -d' ' -f2- \
  | while IFS= read -r b; do sign "$b"; done
sign "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -dv "$APP" 2>&1 | grep -E 'Identifier|Signature|TeamIdentifier|flags' || true

make_dmg() {
  rm -rf dmg && mkdir dmg
  ditto "$APP" "dmg/$(basename "$APP")"
  ln -s /Applications dmg/Applications
  cp "$HERE/macos_first_open.txt" "dmg/首次打开必读.txt"
  hdiutil create -volname PatchClampAnalysis -srcfolder dmg -ov -format UDZO "$NAME.dmg"
  rm -rf dmg
}

notarize() {
  if [ -n "${APPLE_API_KEY_P8_BASE64:-}" ]; then
    echo "$APPLE_API_KEY_P8_BASE64" | base64 --decode > "$RUNNER_TEMP/key.p8"
    xcrun notarytool submit "$1" --wait --timeout 60m \
      --key "$RUNNER_TEMP/key.p8" --key-id "$APPLE_API_KEY_ID" --issuer "$APPLE_API_ISSUER_ID"
    rm -f "$RUNNER_TEMP/key.p8"
  else
    xcrun notarytool submit "$1" --wait --timeout 60m \
      --apple-id "$APPLE_ID" --password "$APPLE_APP_PASSWORD" --team-id "$APPLE_TEAM_ID"
  fi
}

make_dmg
if [ "$IDENTITY" != "-" ]; then
  sign "$NAME.dmg"
fi
if $NOTARIZE; then
  notarize "$NAME.dmg"            # one submission covers the dmg and the app inside it
  xcrun stapler staple "$NAME.dmg"
  xcrun stapler staple "$APP"     # so the update zip carries the ticket too
  spctl -a -vv -t exec "$APP"
fi

# update zip from the final (stapled) app; ditto keeps the signature intact
rm -f "$NAME.zip"
ditto -c -k --keepParent "$APP" "$NAME.zip"

# the zip must unpack into an app with a valid signature
rm -rf zipcheck && mkdir zipcheck
ditto -x -k "$NAME.zip" zipcheck
codesign --verify --deep --strict "zipcheck/$(basename "$APP")"
rm -rf zipcheck
echo "ok: $NAME.zip $NAME.dmg (identity: $IDENTITY, notarized: $NOTARIZE)"
