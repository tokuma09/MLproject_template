import os
from google.oauth2 import service_account

# Mysettings
project_id = "ml-platform-303306"
key_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
credential = service_account.Credentials.from_service_account_file(key_path)
