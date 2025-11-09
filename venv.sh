#!/bin/bash

# Cria o ambiente virtual
python3 -m venv venv

# Ativa o ambiente
source venv/bin/activate

# Atualiza pip
pip install --upgrade pip

# Instala dependências
pip install -r requirements.txt

echo "Fim!"
