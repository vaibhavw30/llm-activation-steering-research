"""Completion-stem prompt sets for the length-steering experiment.

cities: derived programmatically from got_datasets/cities.csv (regular template, reliable to
stem). common_claim: curated, because the free-form claims do not cut into clean stems."""
import re
import pandas as pd

DATASET_DIR = "got_datasets"

# cities rows look like: "The city of Paris is in France."
_CITY_RE = re.compile(r"^The city of (.+?) is in (.+?)\.?$")


def cities_stems(n=300, seed=0):
    df = pd.read_csv(f"{DATASET_DIR}/cities.csv")
    df = df[df["label"] == 1]                      # true statements only
    df = df.sample(min(n, len(df)), random_state=seed)
    out = []
    for s in df["statement"].astype(str):
        m = _CITY_RE.match(s.strip())
        if m:
            city = m.group(1)
            country = m.group(2).strip()
            stem = f"The city of {city} is in the country of"
            out.append((stem, country))
    return out


# ~100 curated (stem, expected_answer) pairs — unambiguous world claims with a crisp answer.
COMMON_CLAIM_STEMS = [
    ("The number of legs a spider has is", "eight"),
    ("The chemical symbol for gold is", "Au"),
    ("The planet known as the Red Planet is", "Mars"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The number of continents on Earth is", "seven"),
    ("The tallest land animal is the", "giraffe"),
    ("The gas humans need to breathe to survive is", "oxygen"),
    ("The organ that pumps blood through the body is the", "heart"),
    ("The freezing point of water in Celsius is", "zero"),
    ("The speed of light is approximately 300,000 kilometers per", "second"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("The currency used in Japan is the", "yen"),
    ("The largest ocean on Earth is the", "Pacific"),
    ("The number of sides a triangle has is", "three"),
    ("The primary language spoken in Brazil is", "Portuguese"),
    ("The metal that is liquid at room temperature is", "mercury"),
    ("The closest star to Earth is the", "Sun"),
    ("The number of days in a leap year is", "366"),
    ("The bone that protects the brain is the", "skull"),
    ("The process plants use to make food is called", "photosynthesis"),
    ("The country that gifted the Statue of Liberty to the US is", "France"),
    ("The hardest natural substance on Earth is", "diamond"),
    ("The number of players on a soccer team on the field is", "eleven"),
    ("The vitamin produced by the skin in sunlight is vitamin", "D"),
    ("The largest mammal on Earth is the blue", "whale"),
    ("The layer of gas that protects Earth from UV rays is the", "ozone"),
    ("The number of strings on a standard guitar is", "six"),
    ("The chemical formula for water is", "H2O"),
    ("The first man to walk on the moon was Neil", "Armstrong"),
    ("The powerhouse of the cell is the", "mitochondria"),
    ("The tallest mountain on Earth is Mount", "Everest"),
    ("The number of colors in a rainbow is", "seven"),
    ("The study of living organisms is called", "biology"),
    ("The largest desert on Earth is the", "Sahara"),
    ("The human body has how many pairs of ribs:", "twelve"),
    ("The inventor of the telephone was Alexander Graham", "Bell"),
    ("The longest river in the world is the", "Nile"),
    ("The number of teeth in a healthy adult human is", "thirty-two"),
    ("The gas that makes up most of Earth's atmosphere is", "nitrogen"),
    ("The country with the largest population is", "India"),
    ("The smallest prime number is", "two"),
    ("The organ responsible for filtering blood is the", "kidney"),
    ("The number of hours in a day is", "twenty-four"),
    ("The painter of the Mona Lisa was Leonardo da", "Vinci"),
    ("The largest internal organ in the human body is the", "liver"),
    ("The number of planets in our solar system is", "eight"),
    ("The primary gas that plants absorb is carbon", "dioxide"),
    ("The continent that is also a country is", "Australia"),
    ("The number of legs an insect has is", "six"),
    ("The unit used to measure electrical resistance is the", "ohm"),
    ("The largest species of shark is the whale", "shark"),
    ("The capital city where the Eiffel Tower stands is", "Paris"),
    ("The frozen form of water is called", "ice"),
    ("The number of zeros in one thousand is", "three"),
    ("The scientist who developed the theory of relativity was Albert", "Einstein"),
    ("The tallest building material grown as grass is", "bamboo"),
    ("The organ used for breathing in fish is the", "gills"),
    ("The number of minutes in an hour is", "sixty"),
    ("The metal most commonly used in electrical wiring is", "copper"),
    ("The natural satellite of Earth is the", "Moon"),
    ("The number of legs a dog has is", "four"),
    ("The largest country by land area is", "Russia"),
    ("The color obtained by mixing blue and yellow is", "green"),
    ("The instrument used to measure temperature is a", "thermometer"),
    ("The number of sides a hexagon has is", "six"),
    ("The first element on the periodic table is", "hydrogen"),
    ("The type of animal a frog is classified as is an", "amphibian"),
    ("The number of legs a horse has is", "four"),
    ("The ship that sank in 1912 after hitting an iceberg was the", "Titanic"),
    ("The number of weeks in a year is", "fifty-two"),
    ("The largest bird in the world is the", "ostrich"),
    ("The gas released by plants during photosynthesis is", "oxygen"),
    ("The number of degrees in a right angle is", "ninety"),
    ("The country where the pyramids of Giza are located is", "Egypt"),
    ("The organ that produces insulin is the", "pancreas"),
    ("The number of players on a basketball team on the court is", "five"),
    ("The nearest planet to the Sun is", "Mercury"),
    ("The word for a baby dog is a", "puppy"),
    ("The number of sides a square has is", "four"),
    ("The largest state in the United States by area is", "Alaska"),
    ("The natural process by which water turns to vapor is", "evaporation"),
    ("The number of legs a spider has is not six but", "eight"),
    ("The metal with the chemical symbol Fe is", "iron"),
    ("The number of months in a year is", "twelve"),
    ("The primary source of energy for Earth is the", "Sun"),
    ("The animal known as man's best friend is the", "dog"),
    ("The number of letters in the English alphabet is", "twenty-six"),
    ("The chemical symbol for oxygen is", "O"),
    ("The country famous for the Great Wall is", "China"),
    ("The organ that allows humans to see is the", "eye"),
    ("The number of seconds in a minute is", "sixty"),
    ("The largest planet's most famous feature is its", "rings"),
    ("The name of the galaxy Earth is in is the Milky", "Way"),
    ("The number of legs an octopus has is", "eight"),
    ("The primary ingredient in bread is", "flour"),
    ("The number of colors on a traffic light is", "three"),
    ("The scientist known for the laws of motion was Isaac", "Newton"),
    ("The frozen continent at the southern pole is", "Antarctica"),
    ("The number of eyes a typical human has is", "two"),
]


def common_claim_stems(n=100):
    return COMMON_CLAIM_STEMS[:n]


def get_prompt_set(dataset, n=None, seed=0):
    if dataset == "cities":
        return cities_stems(n or 300, seed)
    if dataset == "common_claim_true_false":
        return common_claim_stems(n or 100)
    raise ValueError(f"unknown dataset {dataset!r}")
