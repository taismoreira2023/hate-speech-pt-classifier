# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente
.\venv\Scripts\Activate.ps1

# Atualiza pip
pip install --upgrade pip

# Instala dependências
pip install -r requirements.txt

Write-Host "Fim"
