from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import hbp_model

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned model
model = hbp_model.Net()
model.load_state_dict(torch.load("firststep.pth"))
model.eval()

# Define transformation for image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# Define bird species classes
classes = [
    "Black Footed Albatross",
    "Laysan Albatross",
    "Sooty Albatross",
    "Groove Billed Ani",
    "Crested Auklet",
    "Least Auklet",
    "Parakeet Auklet",
    "Rhinoceros Auklet",
    "Brewer Blackbird",
    "Red Winged Blackbird",
    "Rusty Blackbird",
    "Yellow Headed Blackbird",
    "Bobolink",
    "Indigo Bunting",
    "Lazuli Bunting",
    "Painted Bunting",
    "Cardinal",
    "Spotted Catbird",
    "Gray Catbird",
    "Yellow breasted Chat",
    "Eastern Towhee",
    "Chuck Will Widow",
    "Brandt Cormorant",
    "Red Faced Cormorant",
    "Pelagic Cormorant",
    "Bronzed Cowbird",
    "Shiny Cowbird",
    "Brown Creeper",
    "American Crow",
    "Fish Crow",
    "Black Billed Cuckoo",
    "Mangrove Cuckoo",
    "Yellow Billed Cuckoo",
    "Gray Crowned Rosy Finch",
    "Purple Finch",
    "Northern Flicker",
    "Acadian Flycatcher",
    "Great Crested Flycatcher",
    "Least Flycatcher",
    "Olive Sided Flycatcher",
    "Scissor Tailed Flycatcher",
    "Vermilion Flycatcher",
    "Yellow Bellied Flycatcher",
    "Frigatebird",
    "Northern Fulmar",
    "Gadwall",
    "American Goldfinch",
    "European Goldfinch",
    "Boat Tailed Grackle",
    "Eared Grebe",
    "Horned Grebe",
    "Pied Billed Grebe",
    "Western Grebe",
    "Blue Grosbeak",
    "Evening Grosbeak",
    "Pine Grosbeak",
    "Rose Breasted Grosbeak",
    "Pigeon Guillemot",
    "California Gull",
    "Glaucous Winged Gull",
    "Heermann Gull",
    "Herring Gull",
    "Ivory Gull",
    "Ring Billed Gull",
    "Slaty backed Gull",
    "Western Gull",
    "Anna Hummingbird",
    "Ruby Throated Hummingbird",
    "Rufous Hummingbird",
    "Green Violetear",
    "Long Tailed Jaeger",
    "Pomarine Jaeger",
    "Blue Jay",
    "Florida Jay",
    "Green Jay",
    "Dark Eyed Junco",
    "Tropical Kingbird",
    "Gray Kingbird",
    "Belted Kingfisher",
    "Green Kingfisher",
    "Pied Kingfisher",
    "Ringed Kingfisher",
    "White Breasted Kingfisher",
    "Red Legged Kittiwake",
    "Horned Lark",
    "Pacific Loon",
    "Mallard",
    "Western Meadowlark",
    "Hooded Merganser",
    "Red Breasted Merganser",
    "Mockingbird",
    "Nighthawk",
    "Clark Nutcracker",
    "White Breasted Nuthatch",
    "Baltimore Oriole",
    "Hooded Oriole",
    "Orchard Oriole",
    "Scott Oriole",
    "Ovenbird",
    "Brown Pelican",
    "White Pelican",
    "Western Wood Pewee",
    "Sayornis",
    "American Pipit",
    "Whip Poor Will",
    "Horned Puffin",
    "Common Raven",
    "White Necked Raven",
    "American Redstart",
    "Geococcyx",
    "Loggerhead Shrike",
    "Great Grey Shrike",
    "Baird Sparrow",
    "Black Throated Sparrow",
    "Brewer Sparrow",
    "Chipping Sparrow",
    "Clay Colored Sparrow",
    "House Sparrow",
    "Field Sparrow",
    "Fox Sparrow",
    "Grasshopper Sparrow",
    "Harris Sparrow",
    "Henslow Sparrow",
    "Le Conte Sparrow",
    "Lincoln Sparrow",
    "Nelson Sharp Tailed Sparrow",
    "Savannah Sparrow",
    "Seaside Sparrow",
    "Song Sparrow",
    "Tree Sparrow",
    "Vesper Sparrow",
    "White Crowned Sparrow",
    "White Throated Sparrow",
    "Cape Glossy Starling",
    "Bank Swallow",
    "Barn Swallow",
    "Cliff Swallow",
    "Tree Swallow",
    "Scarlet Tanager",
    "Summer Tanager",
    "Artic Tern",
    "Black Tern",
    "Caspian Tern",
    "Common Tern",
    "Elegant Tern",
    "Forsters Tern",
    "Least Tern",
    "Green Tailed Towhee",
    "Brown Thrasher",
    "Sage Thrasher",
    "Black Capped Vireo",
    "Blue Headed Vireo",
    "Philadelphia Vireo",
    "Red Eyed Vireo",
    "Warbling Vireo",
    "White Eyed Vireo",
    "Yellow Throated Vireo",
    "Bay Breasted Warbler",
    "Black And Qhite Warbler",
    "Black Throated Blue Warbler",
    "Blue Winged Warbler",
    "Canada Warbler",
    "Cape May Warbler",
    "Cerulean Warbler",
    "Chestnut Sided Warbler",
    "Golden Winged Warbler",
    "Hooded Warbler",
    "Kentucky Warbler",
    "Magnolia Warbler",
    "Mourning Warbler",
    "Myrtle Warbler",
    "Nashville Warbler",
    "Orange Crowned Warbler",
    "Palm Warbler",
    "Pine Warbler",
    "Prairie Warbler",
    "Prothonotary Warbler",
    "Swainson Warbler",
    "Tennessee Warbler",
    "Wilson Warbler",
    "Worm Eating Warbler",
    "Yellow Warbler",
    "Northern Waterthrush",
    "Louisiana Waterthrush",
    "Bohemian Waxwing",
    "Cedar Waxwing",
    "American Three Toed Woodpecker",
    "Pileated Woodpecker",
    "Red Bellied Woodpecker",
    "Red Cockaded Woodpecker",
    "Red Headed Woodpecker",
    "Downy Woodpecker",
    "Bewick Wren",
    "Cactus Wren",
    "Carolina Wren",
    "House Wren",
    "Marsh Wren",
    "Rock Wren",
    "Winter Wren",
    "Common Yellowthroat",
]


# Function to predict bird species
def predict_bird_species(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Apply transformation
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return classes[predicted.item()]


# Route for home page
@app.route("/")
def home():
    return render_template("index.html")


# Define a dictionary mapping predicted bird species to unique information
species_info = {
    "Black Footed Albatross": """
Length: 68 to 74 cm
Wingspan: 190 to 220 cm
Weight: 2.6 to 4.3 kg
""",
    "Laysan Albatross": """
Length: 81 cm
Wingspan: 195 to 203 cm
Male Weight: 2.4 to 4.1 kg
Female Weight: 1.9 to 3.6 kg

""",
    "Sooty Albatross": """
Length: About 85 cm
Wingspan: 2 meters

""",
    "Groove Billed Ani": """
Length: 34 cm
Weight: 70–90 g
""",
    "Crested Auklet": """
Length: 18–27 cm
Wingspan: 34–50 cm
Weight: 195–330 g
""",
    "Least Auklet": """
Length: 11–27 cm
Wingspan: 38–50 cm
Weight: 225–330 g
""",
    "Parakeet Auklet": """
Length: 20 to 25 cm
Wingspan: 35 to 45 cm
Weight: 150 to 250 grams
""",
    "Rhinoceros Auklet": """
Length: 30 to 35 cm
Wingspan: 45 to 55 cm
Weight: 400 to 600 grams
""",
    "Brewer Blackbird": """
Length: 20 to 25 cm
Wingspan: 35 to 45 cm
Weight: 60 to 90 grams
""",
    "Red Winged Blackbird": """
Length: 18 to 25 cm
Wingspan: 35 to 45 cm
Weight: 40 to 60 grams
""",
    "Rusty Blackbird": """
Length: 20 to 25 cm
Weight: 60 to 90 grams
Wingspan: 35 to 45 cm


""",
    "Yellow Headed Blackbird": """
Length: 20 to 25 cm
Weight: 60 to 90 grams

""",
    "Bobolink": """
Length: 15 to 20 cm
Weight: 30 to 40 grams
""",
    "Indigo Bunting": """
Length: 11 to 13 cm
Weight: 10 to 15 grams
.""",
    "Lazuli Bunting": """
Length: 12 to 15 cm
Weight: 10 to 20 grams
""",
    "Painted Bunting": """
Length: 20 to 33 cm
Weight: 30 to 70 grams
""",
    "Cardinal": """
Length: 20 to 23 cm
Weight: 30 to 50 grams.
""",
    "Spotted Catbird": """
Length: 30 to 35 cm
Wingspan: 45 to 55 cm
Weight: 400 to 600 grams
""",
    "Gray Catbird": """
Length: 20 to 23 cm
Wingspan: 22 to 30 cm
Weight: 23 to 49 grams
""",
    "Yellow Breasted Chat": """
Length: 17 to 19 cm
Wingspan: 21 to 27 cm
Weight: 20 to 25 grams
""",
    "Eastern Towhee": """
Length: 17 to 23 cm
Wingspan: 20 to 30 cm
Weight: 33 to 49 grams
""",
    "Chuck Will Widow": """
Length: 28 to 33 cm
Wingspan: 50 to 60 cm
Weight: 45 to 90 grams
""",
    "Brandt Cormorant": """ 
Length: 70 to 90 cm
Wingspan: 100 to 115 cm
Weight: 1,100 to 1,900 grams
""",
    "Red Faced Cormorant": """
Length: 60 to 70 cm
Wingspan: 90 to 105 cm
Weight: 1,000 to 1,400 grams
""",
    "Pelagic Cormorant": """
Length: 50 to 55 cm
Wingspan: 85 to 95 cm
Weight: 750 to 1,000 grams
""",
    "Bronzed Cowbird": """
Length: 17 to 20 cm
Wingspan: 28 to 33 cm
Weight: 40 to 60 grams
""",
    "Shiny Cowbird": """
Length: 17 to 20 cm
Wingspan: 30 to 34 cm
Weight: 35 to 60 grams
""",
    "Brown Creeper": """
 Length: 12 to 13 cm
Wingspan: 18 to 20 cm
Weight: 7 to 11 grams
""",
    "American Crow": """
Length: 40 to 53 cm
Wingspan: 85 to 100 cm
Weight: 300 to 600 grams
""",
    "Fish Crow": """
Length: 36 to 41 cm
Wingspan: 75 to 85 cm
Weight: 200 to 300 grams
""",
    "Black Billed Cuckoo": """
Length: 27 to 30 cm
Wingspan: 28 to 33 cm
Weight: 45 to 60 grams
""",
    "Mangrove Cuckoo": """"
Length: 28 to 33 cm
Wingspan: 36 to 41 cm
Weight: 50 to 70 grams
""",
    "Yellow Billed Cuckoo": """
Length: 28 to 30 cm
Wingspan: 36 to 41 cm
Weight: 45 to 70 grams
""",
    "Gray Crowned Rosy Finch": """
Length: 15 to 18 cm
Wingspan: 28 to 33 cm
Weight: 20 to 35 grams
""",
    "Purple Finch": """
Length: 15 to 17 cm
Wingspan: 25 to 28 cm
Weight: 20 to 32 grams
""",
    "Northern Flicker": """
Length: 28 to 36 cm
Wingspan: 42 to 54 cm
Weight: 85 to 165 grams
""",
    "Acadian Flycatcher": """
Length: 13 to 14 cm
Wingspan: 20 to 22 cm
Weight: 8 to 12 grams
""",
    "Great Crested Flycatcher": """
Length: 18 to 21 cm
Wingspan: 33 to 38 cm
Weight: 30 to 50 grams

""",
    "Least Flycatcher": """
Length: 12 to 13 cm
Wingspan: 20 to 22 cm
Weight: 7 to 11 grams

""",
    "Olive Sided Flycatcher": """
Length: 18 to 20 cm
Wingspan: 32 to 35 cm
Weight: 28 to 50 grams
""",
    "Scissor Tailed Flycatcher": """
Length: 26 to 38 cm
Wingspan: 38 to 58 cm
Weight: 42 to 95 grams.
""",
    "Vermilion Flycatcher": """
Length: 14 to 15 cm
Wingspan: 22 to 25 cm
Weight: 9 to 12 grams
""",
    "Yellow Bellied Flycatcher": """
Length: 12 to 13 cm
Wingspan: 18 to 20 cm
Weight: 7 to 11 grams
""",
    "Frigatebird": """
Length: 80 to 105 cm
Wingspan: 195 to 230 cm
Weight: 700 to 1,600 grams
""",
    "Northern Fulmar": """
Length: 43 to 51 cm
Wingspan: 102 to 112 cm
Weight: 430 to 900 grams
""",
    "Gadwall": """
Length: 46 to 56 cm
Wingspan: 80 to 90 cm
Weight: 630 to 1,200 grams
""",
    "American Goldfinch": """
Length: 11 to 14 cm
Wingspan: 19 to 22 cm
Weight: 11 to 20 grams
""",
    "European Goldfinch": """
Length: 12 to 13 cm
Wingspan: 21 to 25 cm
Weight: 14 to 18 grams
""",
    "Boat Tailed Grackle": """
Length: 33 to 42 cm
Wingspan: 37 to 48 cm
Weight: 150 to 250 grams
""",
    "Eared Grebe": """
Length: 28 to 36 cm
Wingspan: 40 to 50 cm
Weight: 250 to 500 grams
""",
    "Horned Grebe": "yza",
    "Pied Billed Grebe": "zab",
    "Western Grebe": "abc",
    "Blue Grosbeak": "bcd",
    "Evening Grosbeak": "cde",
    "Pine Grosbeak": "def",
    "Rose Breasted Grosbeak": "efg",
    "Pigeon Guillemot": "fgh",
    "California Gull": "ghi",
    "Glaucous Winged Gull": "hij",
    "Heermann Gull": "ijk",
    "Herring Gull": "jkl",
    "Ivory Gull": "klm",
    "Ring Billed Gull": "lmn",
    "Slaty Backed Gull": "mno",
    "Western Gull": "nop",
    "Anna Hummingbird": "opq",
    "Ruby Throated Hummingbird": "pqr",
    "Rufous Hummingbird": "qrs",
    "Green Violetear": "rst",
    "Long Tailed Jaeger": "stu",
    "Pomarine Jaeger": "tuv",
    "Blue Jay": "uvw",
    "Florida Jay": "vwx",
    "Green Jay": "wxy",
    "Dark Eyed Junco": "xyz",
    "Tropical Kingbird": "yza",
    "Gray Kingbird": "zab",
    "Belted Kingfisher": "abc",
    "Green Kingfisher": "bcd",
    "Pied Kingfisher": "cde",
    "Ringed Kingfisher": "def",
    "White Breasted Kingfisher": "efg",
    "Red Legged Kittiwake": "fgh",
    "Horned Lark": "ghi",
    "Pacific Loon": "hij",
    "Mallard": "ijk",
    "Western Meadowlark": "jkl",
    "Hooded Merganser": "klm",
    "Red Breasted Merganser": "lmn",
    "Mockingbird": "mno",
    "Nighthawk": "nop",
    "Clark Nutcracker": "opq",
    "White Breasted Nuthatch": "pqr",
    "Baltimore Oriole": "qrs",
    "Hooded Oriole": "rst",
    "Orchard Oriole": "stu",
    "Scott Oriole": "tuv",
    "Ovenbird": "uvw",
    "Brown Pelican": """	Length: 106 to 137 cm
Wingspan: 200 to 240 cm
Weight: 2000 to 5000 grams
""",
    "White Pelican": """
Length: 127 to 165 cm
Wingspan: 244 to 290 cm
Weight: 7000 to 9000 grams
""",
    "Western Wood Pewee": """
Length: 13 to 15 cm
Wingspan: 20 to 22 cm
Weight: 10 to 20 grams
""",
    "Sayornis": """
Length: 14 to 17 cm
Wingspan: 22 to 26 cm
Weight: 10 to 20 grams
""",
    "American Pipit": """
Length: 15 to 18 cm
Wingspan: 25 to 30 cm
Weight: 15 to 25 grams
""",
    "Whip Poor Will": """
Length: 19 to 23 cm
Wingspan: 42 to 48 cm
Weight: 40 to 60 grams
""",
    "Horned Puffin": """
Length: 32 to 38 cm
Wingspan: 58 to 65 cm
Weight: 450 to 700 grams
""",
    "Common Raven": """
Length: 54 to 69 cm
Wingspan: 116 to 150 cm
Weight: 600 to 1500 grams
""",
    "White Necked Raven": """Length: 54 to 69 cm
Wingspan: 116 to 150 cm
Weight: 600 to 1500 grams
""",
    "American Redstart": """
Length: 11 to 14 cm
Wingspan: 18 to 22 cm
Weight: 7 to 12 grams
 	
""",
    "Geococcyx": """
Length: 50 to 61 cm
Wingspan: 61 to 66 cm
Weight: 150 to 330 grams
""",
    "Loggerhead Shrike": """
Length: 18 to 23 cm
Wingspan: 28 to 32 cm
Weight: 35 to 50 grams
""",
    "Great Grey Shrike": """Length: 21 to 25 cm
Wingspan: 30 to 35 cm
Weight: 45 to 60 grams
""",
    "Baird Sparrow": """
Length: 12 to 14 cm
Wingspan: 20 to 22 cm
Weight: 10 to 15 grams
""",
    "Black Throated Sparrow": """
Length: 12 to 15 cm
Wingspan: 20 to 23 cm
Weight: 10 to 15 grams
""",
    "Brewer Sparrow": """
Length: 12 to 15 cm
Wingspan: 20 to 23 cm
Weight: 10 to 15 grams
""",
    "Chipping Sparrow": "lmn",
    "Clay Colored Sparrow": "mno",
    "House Sparrow": "nop",
    "Field Sparrow": "opq",
    "Fox Sparrow": "pqr",
    "Grasshopper Sparrow": "qrs",
    "Harris Sparrow": "rst",
    "Henslow Sparrow": "stu",
    "Le Conte Sparrow": "tuv",
    "Lincoln Sparrow": "uvw",
    "Nelson Sharp Tailed Sparrow": "vwx",
    "Savannah Sparrow": "wxy",
    "Seaside Sparrow": "xyz",
    "Song Sparrow": "yza",
    "Tree Sparrow": "zab",
    "Vesper Sparrow": "abc",
    "White Crowned Sparrow": "bcd",
    "White Throated Sparrow": "cde",
    "Cape Glossy Starling": "def",
    "Bank Swallow": "efg",
    "Barn Swallow": "fgh",
    "Cliff Swallow": "ghi",
    "Tree Swallow": "hij",
    "Scarlet Tanager": "ijk",
    "Summer Tanager": "jkl",
    "Artic Tern": "klm",
    "Black Tern": "lmn",
    "Caspian Tern": "mno",
    "Common Tern": "nop",
    "Elegant Tern": "opq",
    "Forsters Tern": "pqr",
    "Least Tern": "qrs",
    "Green Tailed Towhee": "rst",
    "Brown Thrasher": "stu",
    "Sage Thrasher": "tuv",
    "Black Capped Vireo": "uvw",
    "Blue Headed Vireo": "vwx",
    "Philadelphia Vireo": "wxy",
    "Red Eyed Vireo": "xyz",
    "Warbling Vireo": "yza",
    "White Eyed Vireo": "zab",
    "Yellow Throated Vireo": "abc",
    "Bay Breasted Warbler": "bcd",
    "Black And White Warbler": "cde",
    "Black Throated Blue Warbler": "def",
    "Blue Winged Warbler": "efg",
    "Canada Warbler": "fgh",
    "Cape May Warbler": "ghi",
    "Cerulean Warbler": "hij",
    "Chestnut Sided Warbler": "ijk",
    "Golden Winged Warbler": "jkl",
    "Hooded Warbler": "klm",
    "Kentucky Warbler": "lmn",
    "Magnolia Warbler": "mno",
    "Mourning Warbler": "nop",
    "Myrtle Warbler": "opq",
    "Nashville Warbler": "pqr",
    "Orange Crowned Warbler": "qrs",
    "Palm Warbler": "rst",
    "Pine Warbler": "stu",
    "Prairie Warbler": "tuv",
    "Prothonotary Warbler": "uvw",
    "Swainson Warbler": "vwx",
    "Tennessee Warbler": "wxy",
    "Wilson Warbler": "xyz",
    "Worm Eating Warbler": "yza",
    "Yellow Warbler": "zab",
    "Northern Waterthrush": "abc",
    "Louisiana Waterthrush": "bcd",
    "Bohemian Waxwing": "cde",
    "Cedar Waxwing": "def",
    "American Three Toed Woodpecker": "efg",
    "Pileated Woodpecker": "fgh",
    "Red Bellied Woodpecker": "ghi",
    "Red Cockaded Woodpecker": "hij",
    "Red Headed Woodpecker": "ijk",
    "Downy Woodpecker": "jkl",
    "Bewick Wren": "klm",
    "Cactus Wren": "lmn",
    "Carolina Wren": "mno",
    "House Wren": "nop",
    "Marsh Wren": "opq",
    "Rock Wren": "pqr",
    "Winter Wren": "qrs",
    "Common Yellowthroat": "rst",
}


# Route for file upload
@app.route("/static/uploads/", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        filepath = os.path.join("static/uploads", filename)
        f.save(filepath)
        predicted_species = predict_bird_species(filepath)
        return render_template(
            "result.html",
            filename=filename,
            predicted_species=predicted_species,
            species_info=species_info,  # Pass species_info dictionary to the template
        )


if __name__ == "__main__":
    app.run(debug=True)
