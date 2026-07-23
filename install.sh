#!/usr/bin/env bash
# Kokodiv — System Installation & Desktop Shortcut Script
# Run with: sudo ./install.sh

set -e

if [ "$EUID" -ne 0 ]; then
    echo "[!] Please run this installer with sudo: sudo ./install.sh"
    exit 1
fi

INSTALL_DIR="/opt/kokodiv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo " Installing Kokodiv System-Wide..."
echo " Source: $REPO_DIR"
echo " Destination: $INSTALL_DIR"
echo "=================================================="

# 1. Copy repository files to /opt/kokodiv
mkdir -p "$INSTALL_DIR"
cp -rf "$REPO_DIR"/* "$INSTALL_DIR/"

# Make scripts executable
chmod +x "$INSTALL_DIR/scripts/start_kokodiv.sh"
chmod +x "$INSTALL_DIR/scripts/download_models.sh" 2>/dev/null || true

# 2. Create global terminal command `/usr/local/bin/kokodiv`
cat << 'EOF' > /usr/local/bin/kokodiv
#!/usr/bin/env bash
exec /opt/kokodiv/scripts/start_kokodiv.sh "$@"
EOF
chmod +x /usr/local/bin/kokodiv

# 3. Create Linux Desktop Application Shortcut
DESKTOP_FILE="/usr/share/applications/kokodiv.desktop"
cat << EOF > "$DESKTOP_FILE"
[Desktop Entry]
Version=1.1
Type=Application
Name=Kokodiv
Comment=On-Device Multimodal Cocoa Disease Intelligence
Exec=/usr/local/bin/kokodiv
Icon=plant-supporter
Terminal=true
Categories=Agriculture;Science;Education;
EOF

chmod 644 "$DESKTOP_FILE"

# Copy to user's Desktop if available
SUDO_USER_HOME=$(eval echo "~$SUDO_USER")
if [ -d "$SUDO_USER_HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$SUDO_USER_HOME/Desktop/Kokodiv.desktop"
    chmod +x "$SUDO_USER_HOME/Desktop/Kokodiv.desktop" 2>/dev/null || true
    chown "$SUDO_USER:$SUDO_USER" "$SUDO_USER_HOME/Desktop/Kokodiv.desktop" 2>/dev/null || true
fi

echo "=================================================="
echo " ✅ Kokodiv Installation Complete!"
echo " "
echo " You can now launch Kokodiv by:"
echo "   1. Double-clicking the 'Kokodiv' icon on your Desktop"
echo "   2. Typing 'kokodiv' in any terminal"
echo "=================================================="
