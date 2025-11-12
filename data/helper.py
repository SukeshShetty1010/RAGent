# helper.py
import sys
import os
from data.rawg_data import RAWGData
import logging

# Set up basic logging to see what's happening
logging.basicConfig(level=logging.INFO)

def main():
    """
    Main function to initialize RAWGData and fetch a real game name.
    """
    # Check if the API key is set in the environment
    if not os.environ.get("RAWG_API_KEY"):
        print("="*50, file=sys.stderr)
        print("ERROR: RAWG_API_KEY environment variable not set.", file=sys.stderr)
        print("Please get a key from https://rawg.io/apikey", file=sys.stderr)
        print("Then run: export RAWG_API_KEY=\"your_key_here\"", file=sys.stderr)
        print("="*50, file=sys.stderr)
        sys.exit(1) # Exit with an error code

    # Get search query from command line arguments, or use a default
    search_query = "portal"
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])

    print(f"Attempting to find real game matching: '{search_query}'...")

    try:
        # Initialize RAWGData. Since we don't pass a client,
        # it will create its own *real* RAWGClient instance.
        data = RAWGData()
        
        # Call the search method
        results = data.search_and_rank_games(search_query, top_k=1)
        
        if results:
            first_game = results[0]
            print("\n--- Success! ---")
            print(f"Found game: {first_game.get('name')}")
            print(f"Game ID:    {first_game.get('id')}")
            print(f"Released:   {first_game.get('released')}")
            print(f"Rating:     {first_game.get('rating')}")
        else:
            print("\n--- No results found for that query. ---")

    except Exception as e:
        # Catch potential exceptions during API calls
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        logging.exception("Detailed error:")

if __name__ == "__main__":
    main()