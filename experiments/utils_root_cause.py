from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from openpyxl.styles import Alignment
import pickle
import os
import io


def upload_to_google_drive(df, folder_id, output_filename):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    TOKEN_FILE = r'C:\Users\wuhan\.gcp\token.pickle'
    CREDENTIALS_FILE = r'C:\Users\wuhan\.gcp\drive_auth.json'

    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)

    service = build('drive', 'v3', credentials=creds)

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')

    from openpyxl import load_workbook
    buffer.seek(0)
    wb = load_workbook(buffer)
    ws = wb.active
    alignment = Alignment(wrap_text=True, vertical='top', horizontal='left')
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = alignment

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    media = MediaInMemoryUpload(buffer.read(), mimetype=mimetype)

    # check if file already exists in the folder
    query = f"name='{output_filename}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=query, fields='files(id)').execute().get('files', [])

    if existing:
        file_id = existing[0]['id']
        uploaded = service.files().update(fileId=file_id, media_body=media, fields='id').execute()
        print(f"Overwritten file ID: {uploaded.get('id')}")
    else:
        file_metadata = {'name': output_filename, 'parents': [folder_id]}
        uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded file ID: {uploaded.get('id')}")