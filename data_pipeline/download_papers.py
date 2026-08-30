import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

def download_arxiv_papers():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    target_path = os.path.join(raw_dir, "papers.jsonl")

    if os.path.exists(target_path):
        print(f"[Papers Ingestion] Raw papers file already exists at {target_path}")
        return

    print("[Papers Ingestion] Fetching papers from arXiv API...")
    search_queries = ["all:exoplanet", "all:\"habitable zone\"", "all:\"exoplanet atmosphere\""]
    papers = []
    seen_ids = set()

    for query in search_queries:
        print(f"[Papers Ingestion] Querying arXiv with: {query}")
        # Fetch 200 papers per query
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&max_results=200"

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            
            for entry in entries:
                arxiv_url = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
                arxiv_id = arxiv_url.split("/abs/")[-1].split("v")[0]
                
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                title = " ".join(title.split()) if title else ""

                abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text
                abstract = " ".join(abstract.split()) if abstract else ""

                published = entry.find("{http://www.w3.org/2005/Atom}published").text
                year = int(published[:4]) if published else None

                authors = [author.find("{http://www.w3.org/2005/Atom}name").text.strip() 
                           for author in entry.findall("{http://www.w3.org/2005/Atom}author")]

                doi_element = entry.find("{http://arxiv.org/schemas/atom}doi")
                doi = doi_element.text.strip() if doi_element is not None else None

                paper_data = {
                    "paper_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "url": arxiv_url,
                    "topics": ["exoplanets", "astrophysics"],
                    "mentioned_planets": [],
                    "mentioned_stars": []
                }
                papers.append(paper_data)
            
            print(f"[Papers Ingestion] Found {len(entries)} entries for query.")
            time.sleep(1.0) # Respect arXiv API rate limit
        except Exception as e:
            print(f"[Papers Ingestion] Error querying arXiv: {e}")

    # Write to jsonl
    with open(target_path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"[Papers Ingestion] Successfully saved {len(papers)} unique papers to {target_path}")

if __name__ == "__main__":
    download_arxiv_papers()
