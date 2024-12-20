FROM debian:latest

# Mise à jour et installation des dépendances nécessaires pour Rust, curl, wget et unzip
RUN apt-get update && \
    apt-get install -y curl build-essential wget unzip

# Installation Rust avec rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ajout du chemin d'accès de cargo à PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Définir le répertoire de travail
WORKDIR /usr/src/myapp

# Copier les fichiers sources
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY assets ./assets
COPY tests ./tests

# Télécharger et extraire les images de test dans le répertoire assets
RUN wget https://nasext-vaader.insa-rennes.fr/ietr-vaader/moseiik_test_images.zip -P ./assets \
    && unzip -o ./assets/moseiik_test_images.zip -d ./assets \
    && rm ./assets/moseiik_test_images.zip

# Définir l'entrée du conteneur pour exécuter les tests
ENTRYPOINT ["cargo", "test", "--release", "--"]
