import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carrega dataset já limpo
df = pd.read_csv(r'..\dados\dataset_corrigido.csv')

# ---- 1. Seleção de variáveis relevantes ----
# Exemplo: suponha que queremos prever "vegetacao_natural"
# (ajuste conforme o objetivo do seu projeto)
target = 'vegetacao_natural'

# Remove colunas que não ajudam na previsão (IDs, nomes, etc)
df_model = df.drop(columns=['id_municipio', 'soma_partes'], errors='ignore')

# ---- 2. Transformação de variáveis categóricas ----
df_model = pd.get_dummies(df_model, drop_first=True)

# ---- 3. Separar X (features) e y (alvo) ----
X = df_model.drop(columns=[target])
y = df_model[target]

# ---- 4. Normalização das variáveis numéricas ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Converte de volta para DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ---- 5. Salvar dataset preparado ----
df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
df_final.to_csv(r'..\dados\dataset_pronto_para_modelagem.csv', index=False)

print("Dataset preparado e salvo em 'dados/dataset_pronto_para_modelagem.csv'")
