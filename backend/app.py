import os
import psycopg2
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

app = Flask(__name__)

# --- Global Variables ---
MODEL_PATH = "hall_of_fame_model.joblib"
model = None

# Full Hall of Fame Player List
HALL_OF_FAME_PLAYERS = {
    "Kareem Abdul-Jabbar", "Ray Allen", "Carmelo Anthony", "Nate Archibald", "Paul Arizin", "Seimone Augustus",
    "Charles Barkley", "Tom Barlow", "Dick Barnett", "Rick Barry", "Elgin Baylor", "Zelmo Beaty", "John Beckman",
    "Walt Bellamy", "Sergei Belov", "Chauncey Billups", "Dave Bing", "Larry Bird", "Sue Bird", "Carol Blazejowski",
    "Bennie Borgmann", "Chris Bosh", "Sonny Boswell", "Bill Bradley", "Carl Braun", "Joe Brennan", "Roger Brown",
    "Kobe Bryant",
    "Vince Carter", "Swin Cash", "Tamika Catchings", "Al Cervi", "Wilt Chamberlain", "Maurice Cheeks",
    "Zack Clayton", "Nat “Sweetwater” Clifton", "Charles “Tarzan” Cooper", "Chuck Cooper", "Michael Cooper",
    "Cynthia Cooper-Dyke", "Kresimir Cosic", "Bob Cousy", "Dave Cowens", "Joan Crawford", "Billy Cunningham",
    "Denise Curry",
    "Drazen Dalipagic", "Louie Dampier", "Bob Dandridge", "Mel Daniels", "Adrian Dantley", "Bob Davies",
    "Walter Davis", "Forrest DeBernardi", "Dave DeBusschere", "Henry “Dutch” Dehnert", "Vlade Divac",
    "Anne Donovan", "Clyde Drexler", "Joe Dumars", "Tim Duncan",
    "Teresa Edwards", "Paul Endacott", "Alex English", "Julius Erving", "Patrick Ewing",
    "Bud Foster", "Sylvia Fowles", "Walt Frazier", "Marty Friedman", "Joe Fulks",
    "Lauren “Laddie” Gale", "Nick “Nikos” Galis", "Harry Gallatin", "Kevin Garnett", "Pau Gasol",
    "William “Pop” Gates", "George Gervin", "Artis Gilmore", "Manu Ginobili", "Tom Gola", "Gail Goodrich",
    "Hal Greer", "Yolanda Griffith", "Robert Gruenig", "Richie Guerin",
    "Cliff Hagan", "Becky Hammon", "Victor Hanson", "Tim Hardaway", "Lusia Harris-Stewart", "John Havlicek",
    "Connie Hawkins", "Elvin Hayes", "Marques Haynes", "Spencer Haywood", "Tom Heinsohn", "Grant Hill",
    "Nat Holman", "Bob Houbregs", "Dwight Howard", "Bailey Howell", "Lou Hudson", "Chuck Hyatt",
    "John Isaacs", "Dan Issel", "Allen Iverson",
    "Inman Jackson", "Lauren Jackson", "Buddy Jeannette", "Clarence “Fats” Jenkins", "Dennis Johnson",
    "Magic Johnson", "Gus Johnson", "William “Skinny” Johnson", "Neil Johnston", "Bobby Jones", "K.C. Jones",
    "Sam Jones", "Michael Jordan",
    "Jason Kidd", "Bernard King", "Radivoj Korac", "Ed Krause", "Toni Kukoc", "Bob Kurland",
    "Bob Lanier", "Joe Lapchick", "Lisa Leslie", "Nancy Lieberman", "Clyde Lovellette", "Jerry Lucas",
    "Hank Luisetti",
    "Ed Macauley", "Karl Malone", "Moses Malone", "Pete Maravich", "Hortencia Marcari", "Sarunas Marciulionis",
    "Slater Martin", "Bob McAdoo", "Katrina McClain", "Branch McCracken", "Jack McCracken", "Bobby McDermott",
    "George McGinnis", "Tracy McGrady", "Dick McGuire", "Kevin McHale", "Dino Meneghin", "Ann Meyers",
    "George Mikan", "Vern Mikkelsen", "Cheryl Miller", "Reggie Miller", "Yao Ming", "Sidney Moncrief",
    "Earl Monroe", "Maya Moore", "Pearl Moore", "Alonzo Mourning", "Chris Mullin", "Calvin Murphy",
    "Charles “Stretch” Murphy", "Dikembe Mutombo",
    "Steve Nash", "Dirk Nowitzki",
    "Hakeem Olajuwon", "Shaquille O’Neal",
    "Harlan “Pat” Page", "Robert Parish", "Tony Parker", "Gary Payton", "Maciel “Ubiratan” Pereira",
    "Drazen Petrovic", "Bob Pettit", "Andy Phillip", "Paul Pierce", "Scottie Pippen", "Jim Pollard",
    "Cumberland Posey", "Albert “Runt” Pullins",
    "Dino Radja", "Frank Ramsey", "Willis Reed", "Mitch Richmond", "Arnie Risen", "Oscar Robertson",
    "David Robinson", "Guy Rodgers", "Dennis Rodman", "John Roosma", "John “Honey” Russell", "Bill Russell",
    "Arvydas Sabonis", "Ralph Sampson", "Dolph Schayes", "Ernest Schmidt", "Oscar Schmidt", "John Schommer",
    "Charlie Scott", "Barney Sedran", "Uljana Semjonova", "Theresa Shank-Grentz", "Bill Sharman", "Jack Sikma",
    "Katie Smith", "Dawn Staley", "Christian Steinmetz", "John Stockton", "Maurice Stokes", "Sheryl Swoopes",
    "Reece “Goose” Tatum", "Isiah Thomas", "David Thompson", "John “Cat” Thompson", "Tina Thompson",
    "Nate Thurmond", "Michele Timms", "Jack Twyman",
    "Wes Unseld",
    "Robert “Fuzzy” Vandivier",
    "Ed Wachter", "Dwyane Wade", "Chet Walker", "Ben Wallace", "Bill Walton", "Bobby Wanzer", "Ora Washington",
    "Teresa Weatherspoon", "Chris Webber", "Jerry West", "Paul Westphal", "Lindsay Whalen", "Jo Jo White",
    "Nera White", "Lenny Wilkens", "Jamaal Wilkes", "Dominique Wilkins", "Lynette Woodard", "John Wooden",
    "James Worthy",
    "George Yardley"
}

# Define feature columns
FEATURE_COLUMNS = [
    'age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast',
    'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
]

def get_db_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "hall_of_fame_db"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "password"),
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432")
    )
    return conn

@app.route('/')
def home():
    return "Hall of Fame Predictor Backend"

def get_player_data():
    """
    Fetches player data from the database, processes it, and returns a DataFrame.
    """
    try:
        conn = get_db_connection()
        # It's better to explicitly list columns than use '*' in production
        query = "SELECT * FROM player_stats;"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Create the target variable
        df['is_hall_of_famer'] = df['player_name'].apply(lambda x: 1 if x in HALL_OF_FAME_PLAYERS else 0)

        # --- Data Cleaning & Preprocessing ---
        # For now, let's keep it simple.
        # 1. Drop rows with missing values in key statistical columns.
        # A more advanced approach might involve imputation.
        stats_cols = ['gp', 'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']
        df.dropna(subset=stats_cols, inplace=True)

        # 2. Ensure numeric types are correct (read_sql_query usually handles this well, but it's good practice)
        for col in stats_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows that might have had non-numeric data coerced to NaN
        df.dropna(subset=stats_cols, inplace=True)

        return df

    except Exception as e:
        # In a real app, you'd want to log this error
        print(f"Error loading data: {e}")
        return pd.DataFrame()


@app.route('/test-db')
def test_db_connection():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"status": "success", "message": "Database connection successful!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/load-data-test')
def test_load_data():
    df = get_player_data()
    if not df.empty:
        # Return the first 5 rows as JSON to verify
        return jsonify({
            "status": "success",
            "data_head": df.head().to_dict(orient='records'),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "hall_of_fame_counts": df['is_hall_of_famer'].value_counts().to_dict()
        })
    else:
        return jsonify({"status": "error", "message": "Could not load or process data."}), 500


def train_model():
    """
    Trains the Random Forest model and saves it to a file.
    Returns evaluation metrics.
    """
    global model
    print("Loading data for training...")
    df = get_player_data()

    if df.empty:
        print("No data available for training.")
        return None

    # Handle potential infinity values if any
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(subset=FEATURE_COLUMNS, inplace=True)

    if df.shape[0] < 10: # Need enough data to train
        print("Not enough data to train the model after cleaning.")
        return None
    
    if df['is_hall_of_famer'].nunique() < 2:
        print("Training data must contain both Hall of Famers and non-Hall of Famers.")
        return None

    X = df[FEATURE_COLUMNS]
    y = df['is_hall_of_famer']

    # Stratify helps ensure the train/test split has a similar ratio of HOF/non-HOF players
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training model with {len(X_train)} samples...")
    # n_estimators is a key parameter; 100 is a good default.
    # class_weight='balanced' is useful if one class (non-HOF) is much more common than the other.
    temp_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    temp_model.fit(X_train, y_train)

    # Save the newly trained model
    joblib.dump(temp_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Also set the global model variable
    model = temp_model

    # Evaluate the model
    print("Evaluating model...")
    y_pred = temp_model.predict(X_test)
    y_proba = temp_model.predict_proba(X_test)[:, 1] # Probability of class 1

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc_score": roc_auc_score(y_test, y_proba)
    }
    print(f"Evaluation metrics: {metrics}")
    return metrics

def load_model():
    """
    Loads the model from disk. If it doesn't exist, trains a new one.
    """
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"{MODEL_PATH} not found. Training a new model...")
        train_model()
    except Exception as e:
        print(f"Error loading model: {e}. Training a new one.")
        train_model()

@app.route('/train-evaluate', methods=['POST'])
def train_and_evaluate_endpoint():
    """
    An endpoint to manually trigger model training and evaluation.
    """
    metrics = train_model()
    if metrics:
        return jsonify({"status": "success", "message": "Model trained successfully.", "metrics": metrics})
    else:
        return jsonify({"status": "error", "message": "Model training failed."}), 500

@app.route('/predict', methods=['POST'])
def predict_hof_probability():
    global model
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded. Please train the model first."}), 500

    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"status": "error", "message": "No input data provided."}), 400

        # Convert input data to DataFrame, ensuring correct column order and types
        # For a single prediction, it's often easier to create a DataFrame directly
        try:
            player_df = pd.DataFrame([input_data])
            # Ensure all feature columns are present
            if not all(col in player_df.columns for col in FEATURE_COLUMNS):
                missing_cols = [col for col in FEATURE_COLUMNS if col not in player_df.columns]
                return jsonify({"status": "error", "message": f"Missing feature columns: {missing_cols}"}), 400
            
            # Select and order columns as expected by the model
            player_df = player_df[FEATURE_COLUMNS]

            # Ensure data types are correct (e.g., convert to numeric)
            for col in FEATURE_COLUMNS:
                player_df[col] = pd.to_numeric(player_df[col], errors='coerce')
            
            # Check for NaNs after conversion (e.g. if a string was passed for a numeric field)
            if player_df.isnull().values.any():
                 # Find columns with NaNs to provide a more specific error
                nan_cols = player_df.columns[player_df.isnull().any()].tolist()
                return jsonify({"status": "error", 
                                "message": f"Invalid non-numeric data provided for features: {nan_cols}. All features must be numeric."}), 400


        except Exception as e:
            # More specific error for data formatting issues
            return jsonify({"status": "error", "message": f"Error processing input data: {str(e)}"}), 400

        # Predict probability
        # predict_proba returns probabilities for each class: [P(class_0), P(class_1)]
        # We want the probability for class 1 (is_hall_of_famer = 1)
        probability_hof = model.predict_proba(player_df)[0][1] 

        return jsonify({
            "status": "success",
            "player_data": input_data,
            "predicted_probability_hall_of_fame": probability_hof
        })

    except Exception as e:
        # General error catch
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 500


if __name__ == '__main__':
    # Load the model on startup
    load_model()
    # It's good practice to also allow host and port to be set by environment variables for Flask
    port = int(os.environ.get("FLASK_PORT", 5000))
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    app.run(debug=os.environ.get("FLASK_DEBUG", "True").lower() == "true", host=host, port=port)
