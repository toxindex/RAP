{
  description = "RAP minimal devShell, pure pip venv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
        rap-pkg = pkgs.python3Packages.buildPythonPackage {
          pname = "RAP";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = [ pkgs.python3Packages.setuptools ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ python pkgs.git python.pkgs.venvShellHook ];

          shellHook = ''
            export PYTHONPATH=$PWD:$PYTHONPATH
            if [ -f .env ]; then
              echo "Found .env file"
              set -a
              source .env
              set +a
            else
              echo "Warning: .env file not found"
            fi

            if [ ! -d .venv ]; then
              echo "Creating local venv and installing pip packages..."
              python -m venv .venv
              . .venv/bin/activate
              pip install --upgrade pip
              pip install \
                langchain langchain-openai langchain-community langchain-core langgraph agno \
                python-dotenv requests aiohttp typing-extensions idna pubchempy httpx \
                duckduckgo-search pydantic sqlalchemy numpy pandas openai
            else
              echo "Using existing .venv"
              . .venv/bin/activate
            fi

            echo "Python packages in venv:"
            pip list
          '';
        };
        packages.default = rap-pkg;
      }
    );
} 