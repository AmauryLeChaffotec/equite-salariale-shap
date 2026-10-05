"""Génère un fichier RH synthétique : data/employes_exemple.csv

Le salaire est construit à partir de critères objectifs (poste, diplôme, ancienneté,
performance, responsabilités, ville) + du bruit. On y injecte VOLONTAIREMENT :
  - une répartition inégale des postes selon le genre (écart « expliqué »)
  - une pénalité salariale de ~5 % pour les femmes à profil égal (écart « inexpliqué »)
  - quelques cas individuels aberrants (sous-paiement ponctuel, tous genres)
Les données sont fictives : elles servent uniquement à démontrer l'application.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 1500

# poste -> (salaire de base €, département, part des femmes parmi les recrutés, poids de fréquence)
POSTES = {
    "Assistant RH":   (31_000, "RH",         0.80, 8),
    "Comptable":      (36_000, "Finance",    0.65, 9),
    "Commercial":     (40_000, "Ventes",     0.40, 14),
    "Développeur":    (44_000, "IT",         0.25, 18),
    "Data Analyst":   (47_000, "IT",         0.40, 8),
    "Chef de projet": (54_000, "IT",         0.40, 10),
    "Responsable":    (63_000, "Management", 0.35, 7),
    "Directeur":      (92_000, "Direction",  0.25, 2),
}
DIPLOMES = {"Bac": 0.95, "Bac+2": 1.00, "Bac+3": 1.04, "Bac+5": 1.10, "Doctorat": 1.15}
VILLES = {"Paris": 1.08, "Lyon": 1.02, "Nantes": 1.00, "Lille": 0.98, "Toulouse": 1.00}


def main() -> None:
    rng = np.random.default_rng(SEED)

    noms = list(POSTES)
    poids = np.array([POSTES[p][3] for p in noms], dtype=float)
    poste = rng.choice(noms, size=N, p=poids / poids.sum())

    # genre dépendant du poste (ségrégation professionnelle)
    p_femme = np.array([POSTES[p][2] for p in poste])
    genre = np.where(rng.random(N) < p_femme, "F", "H")

    diplome = rng.choice(list(DIPLOMES), size=N, p=[0.10, 0.20, 0.25, 0.38, 0.07])
    ville = rng.choice(list(VILLES), size=N, p=[0.35, 0.20, 0.15, 0.15, 0.15])
    anciennete = np.clip(rng.gamma(shape=2.2, scale=3.5, size=N), 0, 35).round(1)
    age = np.clip(23 + anciennete + rng.normal(4, 4, N), 21, 64).round().astype(int)
    performance = rng.choice([1, 2, 3, 4, 5], size=N, p=[0.04, 0.16, 0.45, 0.28, 0.07])

    est_manager = np.isin(poste, ["Chef de projet", "Responsable", "Directeur"])
    equipe = np.where(est_manager, rng.integers(2, 25, N), 0)
    equipe = np.where(poste == "Directeur", rng.integers(20, 80, N), equipe)

    salaire = np.array([POSTES[p][0] for p in poste], dtype=float)
    salaire = salaire * np.array([DIPLOMES[d] for d in diplome])
    salaire = salaire * (1 + 0.022 * np.minimum(anciennete, 18))          # ancienneté, plafonnée
    salaire = salaire * (1 + 0.035 * (performance - 3))                   # performance
    salaire = salaire * np.array([VILLES[v] for v in ville])
    salaire = salaire + 280 * equipe                                      # responsabilités
    salaire = salaire * rng.normal(1.0, 0.025, N)                         # bruit

    # biais injecté : -5 % pour les femmes à profil égal
    salaire = np.where(genre == "F", salaire * 0.95, salaire)

    # quelques sous-paiements individuels (tous genres)
    anomalies = rng.random(N) < 0.03
    salaire = np.where(anomalies, salaire * rng.uniform(0.84, 0.92, N), salaire)

    df = pd.DataFrame({
        "id_employe": [f"E{i:04d}" for i in range(1, N + 1)],
        "genre": genre,
        "age": age,
        "poste": poste,
        "departement": [POSTES[p][1] for p in poste],
        "niveau_diplome": diplome,
        "anciennete_ans": anciennete,
        "performance": performance,
        "nb_personnes_managees": equipe,
        "ville": ville,
        "salaire_annuel": (salaire / 100).round() * 100,
    })

    out = Path(__file__).parent / "data" / "employes_exemple.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} lignes écrites dans {out}")
    print(df.groupby("genre")["salaire_annuel"].agg(["count", "mean"]).round(0))


if __name__ == "__main__":
    main()
