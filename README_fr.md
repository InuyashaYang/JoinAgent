
# JoinAgent 🚀

<p align="center">
  <img src="./JoiningAI.png" alt="Logo JoinAgent" width="200"/>
</p>

<p align="center">
  <em>Élever les interactions LLM à des hauteurs sans précédent</em>
</p>

<p align="center">
  <a href="README_en.md">English</a> •
  <a href="README.md">中文</a> •
  <a href="README_fr.md">Français</a>
</p>

<p align="center">
  <a href="#caractéristiques-principales">Caractéristiques principales</a> •
  <a href="#installation">Installation</a> •
  <a href="#démarrage-rapide">Démarrage rapide</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#licence">Licence</a>
</p>

---

JoinAgent est un cadre de travail à la pointe de la technologie conçu pour des interactions LLM (Large Language Model) haute performance et concurrentes. Il offre une solution robuste pour les tâches d'IA à grande échelle, avec des capacités avancées d'analyse, de validation et de correction d'erreurs.

## Caractéristiques principales

- 🚄 **Haute concurrence** : Gestion efficace de plusieurs appels LLM simultanément
- 🧠 **Analyse intelligente** : Interprétation et structuration transparentes des sorties LLM
- 🛡️ **Validation et correction d'erreurs** : Assurer l'intégrité des données avec des vérifications et des corrections intégrées
- 🔄 **Système de points de contrôle** : Ne perdez jamais de progrès grâce à notre mécanisme avancé de points de contrôle
- 🔀 **Incitation flexible** : Prise en charge des modèles d'invite uniques et multiples
- ⏱️ **Gestion du temps** : Délais d'attente intégrés pour gérer les tâches sans réponse

## Installation

```bash
pip install joinagent
```

## Démarrage rapide

```python
from JoinAgent import MultiProcessor, MultiLLM, LLMParser, TextDivider

# Configurer l'environnement et le chemin du fichier
file_path = 'votre_chemin_de_fichier'

# Initialiser les composants
llm = MultiLLM()
parser = LLMParser()
divider = TextDivider(threshold=4096, overlap=128)

# Définir les modèles et les invites
data_template = '''
{"pos1":['objet mathématique 1','objet mathématique 2',...]}
'''

prompt_template = '''
Vous êtes un assistant méticuleux. Je vais vous donner un texte provenant de documents mathématiques, veuillez m'aider à extraire tous les objets mathématiques entités du texte et les placer dans une liste unifiée.
Pendant votre travail, vous désactiverez complètement la fonction de recherche et les connexions aux sources externes, en vous appuyant uniquement sur le contenu du texte lui-même pour accomplir cette tâche. N'ajoutez pas arbitrairement de nouveaux objets mathématiques.
Si vous extrayez un objet mathématique mais que vous ne trouvez pas sa définition dans le texte, veuillez ne pas produire cet objet mathématique.
Veuillez ne rien ajouter d'autre au début ou à la fin de votre sortie, à l'exception de cette liste.
Le format d'extraction est :
{data_template}

Note spéciale : Lorsque vous rencontrez des exemples, des exercices, des problèmes pratiques, etc. dans le texte, veuillez les ignorer directement et ne pas analyser leur contenu !!
Les nombres, les formules mathématiques, les expressions algébriques, les lettres et autres contenus sans caractères chinois ne sont pas considérés comme des objets mathématiques, veuillez les supprimer !
Veuillez ne pas produire d'objets mathématiques référentiels comme "fonction f", "matrice B" qui ne sont pas définis universellement mais seulement définis dans le contexte.

Voici le texte que je vous donne : {pos1}, veuillez m'aider à extraire les objets mathématiques et les placer dans une liste.
'''

correction_prompt = '''
Vous êtes un correcteur rigoureux. Je vais vous donner une structure de données générée par un grand modèle, veuillez la relire et la corriger selon le format et le contenu spécifiés.

Le format de relecture est :
{data_template}

Voici le texte à vérifier : {answer}, veuillez m'aider à relire et corriger cette liste.
'''

def validation(text):
    return True

# Créer une instance de MultiProcessor
processor = MultiProcessor(llm=llm, 
                           parse_method=parser.parse_dict, 
                           data_template=data_template, 
                           prompt_template=prompt_template, 
                           correction_template=correction_prompt, 
                           validator=validation)

# Traiter le texte
text_list = divider.divide(file_path)
text_dict = {index: {"pos1": value} for index, value in enumerate(text_list)}

# Exécuter le traitement multi-tâches
results = processor.multitask_perform(text_dict, num_threads=5)

# Imprimer les résultats
for index, result in results.items():
    print(f"Morceau {index}: {result}")

```

## Documentation

Pour une documentation complète, veuillez visiter notre [site de documentation officiel](https://docs.joinagent.ai).


---

<p align="center">
  Conçu avec ❤️ par <a href="https://github.com/InuyashaYang">JoiningAI</a>
</p>