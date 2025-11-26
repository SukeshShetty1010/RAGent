{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "file:///mnt/data/merged_game_schema_corrected.json",
  "title": "MergedGame",
  "description": "Unified JSON schema for merged RAWG + IGDB + GameSpot game records.",
  "type": "object",
  "required": [
    "title"
  ],
  "additionalProperties": false,
  "properties": {
    "unified_id": {
      "type": [
        "string",
        "null"
      ],
      "description": "Canonical merged id (e.g. 'rawg:12345' or 'igdb:5678')."
    },
    "title": {
      "type": "string",
      "minLength": 1,
      "description": "Canonical game title (non-empty)."
    },
    "slug": {
      "type": [
        "string",
        "null"
      ],
      "description": "Canonical slug (prefer IGDB or computed)."
    },
    "description": {
      "type": [
        "string",
        "null"
      ],
      "description": "Long-form description (HTML stripped where applicable)."
    },
    "release_date": {
      "type": [
        "string",
        "null"
      ],
      "format": "date",
      "description": "Primary release date (ISO format)."
    },
    "release_year": {
      "type": [
        "integer",
        "null"
      ],
      "minimum": 1950,
      "maximum": 2100
    },
    "release_dates": {
      "type": "array",
      "description": "List of release date records: platform, region, date, source.",
      "items": {
        "type": "object",
        "properties": {
          "platform": {
            "type": [
              "string",
              "null"
            ]
          },
          "region": {
            "type": [
              "string",
              "null"
            ]
          },
          "date": {
            "type": [
              "string",
              "null"
            ],
            "format": "date"
          },
          "source": {
            "type": [
              "string",
              "null"
            ]
          },
          "notes": {
            "type": [
              "string",
              "null"
            ]
          }
        },
        "additionalProperties": false
      }
    },
    "rawg_id": {
      "type": [
        "integer",
        "null"
      ]
    },
    "igdb_id": {
      "type": [
        "integer",
        "null"
      ]
    },
    "gamespot_id": {
      "type": [
        "string",
        "null"
      ]
    },
    "platforms": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "genres": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "tags": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "themes": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "developers": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "publishers": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "age_ratings": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "esrb_rating": {
      "type": [
        "string",
        "null"
      ]
    },
    "ratings": {
      "type": "object",
      "properties": {
        "rawg": {
          "type": [
            "number",
            "null"
          ]
        },
        "igdb": {
          "type": [
            "number",
            "null"
          ]
        },
        "metacritic": {
          "type": [
            "integer",
            "null"
          ]
        },
        "rawg_detail": {
          "description": "RAWG rating breakdown; can be list or dict.",
          "anyOf": [
            {
              "type": "object",
              "additionalProperties": true
            },
            {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": true
              }
            }
          ]
        },
        "igdb_detail": {
          "description": "IGDB rating detail; can be list or dict.",
          "anyOf": [
            {
              "type": "object",
              "additionalProperties": true
            },
            {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": true
              }
            }
          ]
        },
        "normalized_0_100": {
          "type": [
            "integer",
            "null"
          ],
          "minimum": 0,
          "maximum": 100
        }
      },
      "additionalProperties": false
    },
    "urls": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string",
        "format": "uri"
      }
    },
    "stores": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string"
      }
    },
    "websites": {
      "type": "array",
      "uniqueItems": true,
      "items": {
        "type": "string",
        "format": "uri"
      }
    },
    "gamespot": {
      "type": "object",
      "properties": {
        "game_info": {
          "type": "object",
          "additionalProperties": true
        },
        "articles": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": true
          }
        },
        "related": {
          "type": "object",
          "properties": {
            "articles": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": true
              }
            },
            "reviews": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": true
              }
            },
            "releases": {
              "type": "array",
              "items": {
                "type": "object",
                "additionalProperties": true
              }
            }
          },
          "additionalProperties": true
        }
      },
      "additionalProperties": true
    },
    "documents": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "integer" },
          "source": { "type": "string" },
          "title": { "type": "string" },
          "content": { "type": "string" },
          "excerpt": { "type": "string" },
          "created_at": { "type": "string", "format": "date" },
          "meta": { "type": "object", "additionalProperties": true }
        },
        "required": ["id", "source", "title", "content"]
      }
    },
     "source": {
      "type": "object",
      "additionalProperties": true
    },
    "merged_from": {
        "type": "object",
        "properties": {
            "title_from": { "type": "string" },
            "description_from": { "type": "string" },
            "release_from": { "type": "string" }
        }
    }
  }
}