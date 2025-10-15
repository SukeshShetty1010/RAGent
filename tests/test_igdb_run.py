from api.igdb_client import igdb_request

games = igdb_request("games", "fields name,genres.name; limit 3;")
print(games)