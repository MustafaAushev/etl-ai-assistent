import os
import tempfile
from typing import Annotated
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.responses import JSONResponse
import uvicorn
from services.chunk_splitter_service import ChunkSplitterService
from services.qdrant_service import QdrantService
from parsers.docx_parser import DocxParser

qdrant = QdrantService()
app = FastAPI()

from fastapi import Form

@app.post('/api/v1/upload_docx')
async def upload_docx(
    file: Annotated[UploadFile, Form()], 
    version: Annotated[str, Form()]
):

    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are allowed")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            file_content = await file.read()
            tmp.write(file_content)
            tmp_filename = tmp.name
        
        
        
        document = DocxParser.parse(file.filename, tmp_filename)
        os.remove(tmp_filename)

        document = ChunkSplitterService.make_chunks(document)

        qdrant.load_items_to_collection(document, version)

        return JSONResponse(content={'success': 'ok'})
    except Exception as exc:
        print(exc)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(exc)}") from exc

if __name__ == "__main__":
  uvicorn.run("app:app", host="0.0.0.0", port=8000)
