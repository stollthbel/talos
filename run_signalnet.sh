#!/bin/bash

# Install OCaml if not present
if ! command -v ocaml &> /dev/null; then
    echo "Installing OCaml..."
    sudo apt update && sudo apt install -y ocaml
fi

# Navigate to OCaml directory
cd /workspaces/talos/OCaml

# Run the SignalNet system
echo "🔥 Running SignalNet Evolution System 🔥"
ocaml signalnet.ml
