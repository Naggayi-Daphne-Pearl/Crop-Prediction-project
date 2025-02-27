from google.oauth2 import service_account
from googleapiclient.discovery import build
import time
import json

class SheetsLogger:
    def __init__(self, credentials_file, spreadsheet_id):
        self.SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        self.SPREADSHEET_ID = spreadsheet_id
        
        # Load credentials
        self.creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=self.SCOPES)
        
        # Create sheets service
        self.service = build('sheets', 'v4', credentials=self.creds)
        self.sheet = self.service.spreadsheets()
    
    def log_results(self, results):
        """
        Log results to Google Sheets
        results: dict containing the model metrics
        """
        # Prepare the data row
        values = [[
            results.get('model_name', ''),
            results.get('teacher_training_rate', ''),
            results.get('teacher_learning_rate', ''),
            results.get('teacher_training_accuracy', ''),
            results.get('student_training_rate', ''),
            results.get('student_learning_rate', ''),
            results.get('student_training_accuracy', '')
        ]]
        
        # Append the row to the sheet
        body = {
            'values': values
        }
        
        try:
            result = self.sheet.values().append(
                spreadsheetId=self.SPREADSHEET_ID,
                range='Sheet1!A2',  # Start from A2 to preserve headers
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            print(f"Results logged successfully: {result}")
            return True
        except Exception as e:
            print(f"Error logging to sheets: {e}")
            return False 