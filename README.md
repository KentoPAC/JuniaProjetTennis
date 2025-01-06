# 🎾 JuniaProjetTennis

Bienvenue dans le repo du projet **Tennis** de **JUNIA** !

Ce projet utilise **Pre-commit** pour automatiser le formatage du code et garantir un code propre et cohérent. Nous te remercions d'installer **Pre-commit** avant de commencer à contribuer.

## 🚀 Installation de Pre-commit

Avant de commencer à coder, assure-toi d'installer **Pre-commit** sur ton environnement local. Suis les étapes ci-dessous pour l'installation.

1. **Installation de Pre-commit** :
   - Pour installer Pre-commit, tu peux utiliser **pip** :
     ```bash
     pip install pre-commit
     ```

2. **Configurer Pre-commit pour ce projet** :
   - Une fois **Pre-commit** installé, exécute la commande suivante dans le répertoire du projet :
     ```bash
     pre-commit install
     ```
   - Cela installera tous les hooks définis dans le fichier `.pre-commit-config.yaml`.

3. **Exécuter les hooks manuellement (facultatif)** :
   - Si tu souhaites exécuter les hooks manuellement à tout moment, utilise cette commande :
     ```bash
     pre-commit run --all-files
     ```

> 📖 Pour plus d'infos, consulte la documentation officielle de **Pre-commit** : [https://pre-commit.com/](https://pre-commit.com/)

---

## 📝 Description du projet

Le projet **JuniaProjetTennis** consiste à développer un système d'analyse de matchs de tennis. Ce projet vise à améliorer les performances des joueurs en fournissant des statistiques et des informations sur leurs matchs.

### Fonctionnalités principales :
- Suivi des coups de raquette en temps réel.
- Comptage des points et affichage des statistiques du joueur.
- Application mobile connectée pour visualiser les résultats et les analyses.

---

## 💻 Prérequis

Assure-toi que tu as les outils suivants installés avant de commencer à développer sur ce projet :

- Python 3.10+
- **Pre-commit** installé (voir section ci-dessus)
- Outils de développement en C++ (si nécessaire pour certaines fonctionnalités du projet)

---

## 🔧 Installation

Pour cloner le projet et installer les dépendances, suis les étapes ci-dessous :

1. **Cloner le repo** :
Après avoir mis la clé ssh publique :
   ```bash
   git clone git@github.com:KentoPAC/JuniaProjetTennis.git
   cd JuniaProjetTennis
