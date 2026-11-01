"""
Document processing utilities for medical data ingestion
"""
import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import PyPDF2
from docx import Document
import requests
from bs4 import BeautifulSoup


class MedicalDocument:
    """Represents a medical document with metadata"""
    
    def __init__(
        self,
        content: str,
        title: str,
        source: str,
        doc_id: Optional[str] = None,
        year: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.title = title
        self.source = source
        self.doc_id = doc_id or self._generate_doc_id(content, title)
        self.year = year
        self.url = url
        self.metadata = metadata or {}
        self.processed_at = datetime.now().isoformat()
    
    def _generate_doc_id(self, content: str, title: str) -> str:
        """Generate unique document ID from content and title"""
        hash_input = f"{title}:{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "year": self.year,
            "url": self.url,
            "metadata": self.metadata,
            "processed_at": self.processed_at
        }


class DocumentLoader:
    """Handles loading documents from various sources"""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[str]:
        """Extract text from PDF file"""
        text_chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_chunks.append(text)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return text_chunks
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return ""
    
    @staticmethod
    def load_json(file_path: str) -> List[Dict[str, Any]]:
        """Load structured data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading JSON {file_path}: {e}")
            return []


class PubMedLoader:
    """Fetch medical abstracts from PubMed"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: str = "your_email@example.com"):
        self.email = email
    
    def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        search_url = f"{self.BASE_URL}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            time.sleep(0.4)  # stay under NCBI's ~3 requests/sec limit (no API key)
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            time.sleep(0.4)
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[MedicalDocument]:
        """Fetch abstracts for given PMIDs"""
        if not pmids:
            return []
        
        fetch_url = f"{self.BASE_URL}efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }
        
        documents = []
        try:
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            time.sleep(0.4)  # stay under NCBI's ~3 requests/sec limit (no API key)
            soup = BeautifulSoup(response.content, 'xml')
            
            articles = soup.find_all('PubmedArticle')
            for article in articles:
                try:
                    pmid = article.find('PMID').text
                    title_elem = article.find('ArticleTitle')
                    title = title_elem.text if title_elem else "No title"
                    
                    abstract_elem = article.find('AbstractText')
                    abstract = abstract_elem.text if abstract_elem else ""
                    
                    year_elem = article.find('PubDate')
                    year = None
                    if year_elem and year_elem.find('Year'):
                        year = year_elem.find('Year').text
                    
                    if abstract:
                        doc = MedicalDocument(
                            content=abstract,
                            title=title,
                            source="PubMed",
                            doc_id=f"PMID_{pmid}",
                            year=year,
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            metadata={"pmid": pmid}
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
        
        except Exception as e:
            print(f"Error fetching abstracts: {e}")
        
        return documents


class WHOGHOLoader:
    """
    Fetch authentic health indicator data from WHO's official
    Global Health Observatory (GHO) OData API — no API key required.

    Docs: https://www.who.int/data/gho/info/gho-odata-api
    Base endpoint: https://ghoapi.azureedge.net/api/{IndicatorCode}

    The API returns raw numeric rows (country, year, value). Since a RAG
    chatbot needs prose to cite, each row is converted into a short,
    factual sentence before being wrapped as a MedicalDocument.
    """

    BASE_URL = "https://ghoapi.azureedge.net/api"

    # A curated set of well-known, high-value indicator codes so the UI
    # doesn't require memorizing WHO's indicator codebook.
    COMMON_INDICATORS = {
        "WHOSIS_000001": "Life expectancy at birth (years)",
        "WHOSIS_000002": "Healthy life expectancy (HALE) at birth (years)",
        "NCD_BMI_30A": "Prevalence of obesity among adults (%)",
        "M_Est_tob_curr": "Prevalence of current tobacco use (%)",
        "SA_0000001688": "Alcohol, total per capita consumption (litres of pure alcohol)",
        "MDG_0000000001": "Infant mortality rate (per 1000 live births)",
        "MDG_0000000026": "Maternal mortality ratio (per 100,000 live births)",
        "WHS4_100": "Measles immunization coverage among 1-year-olds (%)",
        "HIV_0000000001": "Estimated number of people living with HIV",
        "MALARIA_EST_INCIDENCE": "Estimated malaria incidence (per 1000 population at risk)",
        "TB_e_inc_100k": "Estimated tuberculosis incidence (per 100,000 population)",
        "NCD_GLUC_04": "Prevalence of diabetes among adults (%)",
        "SDGPM25": "Ambient air pollution — mean annual PM2.5 exposure (µg/m³)",
    }

    def fetch_indicator(
        self,
        indicator_code: str,
        country_iso3: Optional[str] = None,
        latest_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query a single indicator, optionally filtered by ISO3 country code."""
        url = f"{self.BASE_URL}/{indicator_code}"
        params: Dict[str, Any] = {"$top": latest_n}
        if country_iso3:
            params["$filter"] = f"SpatialDim eq '{country_iso3.upper()}'"

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            print(f"Error fetching WHO GHO indicator {indicator_code}: {e}")
            return []

    def to_documents(
        self,
        indicator_code: str,
        indicator_label: Optional[str] = None,
        country_iso3: Optional[str] = None,
        latest_n: int = 50,
    ) -> List[MedicalDocument]:
        """Fetch an indicator and convert each row into a citeable MedicalDocument."""
        rows = self.fetch_indicator(indicator_code, country_iso3, latest_n)
        label = indicator_label or self.COMMON_INDICATORS.get(indicator_code, indicator_code)

        # Group rows by country so each document reads as a short country profile
        # instead of one document per single data point.
        by_country: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            country = row.get("SpatialDim", "UNKNOWN")
            by_country.setdefault(country, []).append(row)

        documents = []
        for country, country_rows in by_country.items():
            country_rows.sort(key=lambda r: r.get("TimeDim", 0), reverse=True)
            lines = [f"WHO indicator '{label}' ({indicator_code}) for {country}:"]
            for row in country_rows[:10]:
                year = row.get("TimeDim", "n/a")
                value = row.get("NumericValue", row.get("Value", "n/a"))
                sex = row.get("Dimensions", {}).get("SEX") if isinstance(row.get("Dimensions"), dict) else None
                sex_str = f" ({sex})" if sex else ""
                lines.append(f"- In {year}, the value was {value}{sex_str}.")
            content = "\n".join(lines)

            documents.append(
                MedicalDocument(
                    content=content,
                    title=f"{label} — {country}",
                    source="WHO Global Health Observatory (GHO)",
                    year=str(country_rows[0].get("TimeDim", "")) if country_rows else None,
                    url=f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{indicator_code}",
                    metadata={"indicator_code": indicator_code, "country": country},
                )
            )
        return documents


class MedicalCorpusBuilder:
    """Build and manage medical document corpus"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw_documents"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_local_documents(self) -> List[MedicalDocument]:
        """Load documents from local directory"""
        documents = []
        loader = DocumentLoader()
        
        for file_path in self.raw_dir.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                
                try:
                    if suffix == ".pdf":
                        pages = loader.load_pdf(str(file_path))
                        for i, page_text in enumerate(pages):
                            doc = MedicalDocument(
                                content=page_text,
                                title=f"{file_path.stem} - Page {i+1}",
                                source=file_path.name,
                                metadata={"file_path": str(file_path), "page": i+1}
                            )
                            documents.append(doc)
                    
                    elif suffix == ".docx":
                        content = loader.load_docx(str(file_path))
                        if content:
                            doc = MedicalDocument(
                                content=content,
                                title=file_path.stem,
                                source=file_path.name,
                                metadata={"file_path": str(file_path)}
                            )
                            documents.append(doc)
                    
                    elif suffix == ".txt":
                        content = loader.load_txt(str(file_path))
                        if content:
                            doc = MedicalDocument(
                                content=content,
                                title=file_path.stem,
                                source=file_path.name,
                                metadata={"file_path": str(file_path)}
                            )
                            documents.append(doc)
                    
                    elif suffix == ".json":
                        json_data = loader.load_json(str(file_path))
                        for item in json_data:
                            if isinstance(item, dict) and "content" in item:
                                doc = MedicalDocument(
                                    content=item.get("content", ""),
                                    title=item.get("title", file_path.stem),
                                    source=item.get("source", file_path.name),
                                    year=item.get("year"),
                                    url=item.get("url"),
                                    metadata=item.get("metadata", {})
                                )
                                documents.append(doc)
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return documents
    
    def save_corpus(self, documents: List[MedicalDocument], filename: str = "corpus.json"):
        """Save processed corpus to JSON file"""
        corpus_path = self.processed_dir / filename
        corpus_data = [doc.to_dict() for doc in documents]
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(documents)} documents to {corpus_path}")
        return corpus_path
    
    def load_corpus(self, filename: str = "corpus.json") -> List[MedicalDocument]:
        """Load processed corpus from JSON file"""
        corpus_path = self.processed_dir / filename
        
        if not corpus_path.exists():
            return []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        documents = []
        for data in corpus_data:
            doc = MedicalDocument(
                content=data["content"],
                title=data["title"],
                source=data["source"],
                doc_id=data["doc_id"],
                year=data.get("year"),
                url=data.get("url"),
                metadata=data.get("metadata", {})
            )
            documents.append(doc)
        
        return documents


class MedlinePlusLoader:
    """
    Fetch authoritative patient-education articles from the U.S. National Library of
    Medicine's MedlinePlus Connect API — completely free, no API key required.

    API docs: https://medlineplus.gov/connect/service.html
    Endpoint:  https://connect.medlineplus.gov/application?
                   mainSearchCriteria.v.cs=2.16.840.1.113883.6.90   (ICD-10)
                   &mainSearchCriteria.v.c=<ICD10_CODE>
                   &knowledgeResponseType=application/json

    Alternatively use the full-text search endpoint:
        https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=<QUERY>
    which returns NLM HealthTopics XML that we parse into MedicalDocuments.

    Usage example
    -------------
    loader = MedlinePlusLoader()
    docs = loader.search("anemia")          # free-text search
    docs += loader.fetch_by_icd10("D50")    # ICD-10 lookup (iron-def anaemia)
    """

    SEARCH_URL = "https://wsearch.nlm.nih.gov/ws/query"
    CONNECT_URL = "https://connect.medlineplus.gov/application"

    # ICD-10-CM coding system OID used by the Connect API
    ICD10_CS = "2.16.840.1.113883.6.90"

    def search(self, query: str, max_results: int = 10) -> List[MedicalDocument]:
        """Full-text search of MedlinePlus Health Topics."""
        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": max_results,
            "rettype": "brief",
        }
        try:
            resp = requests.get(self.SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            return self._parse_nlm_xml(resp.content, query)
        except Exception as e:
            print(f"MedlinePlus search error for '{query}': {e}")
            return []

    def fetch_by_icd10(self, icd10_code: str) -> List[MedicalDocument]:
        """Look up MedlinePlus articles by ICD-10 code via the Connect API."""
        params = {
            "mainSearchCriteria.v.cs": self.ICD10_CS,
            "mainSearchCriteria.v.c": icd10_code.upper(),
            "knowledgeResponseType": "application/json",
        }
        try:
            resp = requests.get(self.CONNECT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_connect_json(data, icd10_code)
        except Exception as e:
            print(f"MedlinePlus Connect error for ICD-10 '{icd10_code}': {e}")
            return []

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    def _parse_nlm_xml(self, xml_bytes: bytes, query: str) -> List[MedicalDocument]:
        """Parse NLM HealthTopics XML search results."""
        docs = []
        try:
            soup = BeautifulSoup(xml_bytes, "xml")
            for doc_elem in soup.find_all("document"):
                url = doc_elem.get("url", "")
                title_elem = doc_elem.find("content", {"name": "title"})
                snippet_elem = doc_elem.find("content", {"name": "FullSummary"})
                if not snippet_elem:
                    snippet_elem = doc_elem.find("content", {"name": "snippet"})

                title = title_elem.get_text(strip=True) if title_elem else query
                content = snippet_elem.get_text(separator=" ", strip=True) if snippet_elem else ""
                content = " ".join(content.split())  # normalise whitespace

                if content and len(content) > 80:
                    docs.append(MedicalDocument(
                        content=content,
                        title=f"MedlinePlus: {title}",
                        source="MedlinePlus (NLM/NIH)",
                        url=url,
                        metadata={"query": query, "api": "nlm_healthtopics"},
                    ))
        except Exception as e:
            print(f"MedlinePlus XML parse error: {e}")
        return docs

    def _parse_connect_json(self, data: dict, icd10_code: str) -> List[MedicalDocument]:
        """Parse MedlinePlus Connect JSON response."""
        docs = []
        try:
            feed = data.get("feed", {})
            entries = feed.get("entry", [])
            for entry in entries:
                title = entry.get("title", {}).get("_value", icd10_code)
                summary = entry.get("summary", {}).get("_value", "")
                url = ""
                for link in entry.get("link", []):
                    if link.get("@type") == "text/html":
                        url = link.get("@href", "")
                        break
                content = BeautifulSoup(summary, "html.parser").get_text(separator=" ", strip=True)
                content = " ".join(content.split())
                if content and len(content) > 80:
                    docs.append(MedicalDocument(
                        content=content,
                        title=f"MedlinePlus: {title}",
                        source="MedlinePlus (NLM/NIH)",
                        url=url,
                        metadata={"icd10_code": icd10_code, "api": "medlineplus_connect"},
                    ))
        except Exception as e:
            print(f"MedlinePlus Connect JSON parse error: {e}")
        return docs


class OpenFDADrugLoader:
    """
    Fetch drug label information from the U.S. FDA's openFDA Drug Label API.
    Free, no API key required (rate-limited at 240 requests/minute without a key,
    or 1000/minute with a free key obtained from https://open.fda.gov/apis/authentication/).

    API docs: https://open.fda.gov/apis/drug/label/
    Endpoint:  https://api.fda.gov/drug/label.json?search=<QUERY>&limit=<N>

    Each label section (indications, warnings, dosage, adverse reactions, etc.) is
    converted into a separate MedicalDocument so retrieval targets the right section.

    Usage example
    -------------
    loader = OpenFDADrugLoader()
    docs = loader.search_drug("metformin", max_results=3)
    """

    BASE_URL = "https://api.fda.gov/drug/label.json"

    # Label sections we extract — ordered from most to least patient-relevant.
    # Keys match openFDA's field names in the label JSON.
    SECTIONS = [
        ("indications_and_usage", "Indications and Usage"),
        ("warnings", "Warnings"),
        ("warnings_and_cautions", "Warnings and Cautions"),
        ("adverse_reactions", "Adverse Reactions"),
        ("drug_interactions", "Drug Interactions"),
        ("contraindications", "Contraindications"),
        ("dosage_and_administration", "Dosage and Administration"),
        ("mechanism_of_action", "Mechanism of Action"),
        ("clinical_pharmacology", "Clinical Pharmacology"),
        ("overdosage", "Overdosage"),
        ("use_in_specific_populations", "Use in Specific Populations"),
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # optional — see class docstring

    def search_drug(self, drug_name: str, max_results: int = 5) -> List[MedicalDocument]:
        """Search openFDA drug labels for a given drug name or active ingredient."""
        # Search both brand name and generic name fields
        search_query = (
            f'(openfda.brand_name:"{drug_name}"+openfda.generic_name:"{drug_name}")'
        )
        params: Dict[str, Any] = {
            "search": search_query,
            "limit": max_results,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            time.sleep(0.25)  # polite rate limiting
        except Exception as e:
            print(f"openFDA drug search error for '{drug_name}': {e}")
            return []

        docs = []
        for label in results:
            docs.extend(self._label_to_documents(label, drug_name))
        return docs

    def _label_to_documents(self, label: dict, query: str) -> List[MedicalDocument]:
        """Convert one FDA label JSON record into multiple MedicalDocuments by section."""
        docs = []
        openfda = label.get("openfda", {})

        # Prefer generic name, fall back to brand name, then query
        generic_names = openfda.get("generic_name", [])
        brand_names = openfda.get("brand_name", [])
        drug_name = generic_names[0].title() if generic_names else (
            brand_names[0].title() if brand_names else query.title()
        )
        manufacturer = openfda.get("manufacturer_name", ["Unknown"])[0]
        application_numbers = openfda.get("application_number", [])
        url = (
            f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo="
            + application_numbers[0].lstrip("NDA").lstrip("ANDA")
            if application_numbers else "https://www.fda.gov/drugs"
        )

        for field_key, section_title in self.SECTIONS:
            texts = label.get(field_key, [])
            if not texts:
                continue
            # openFDA returns sections as a list of strings (usually one entry)
            content = " ".join(texts)
            # Strip residual HTML tags that some labels contain
            content = BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)
            content = " ".join(content.split())
            if len(content) < 50:
                continue
            # Truncate very long sections to avoid noisy chunks
            if len(content) > 3000:
                content = content[:3000] + "..."

            docs.append(MedicalDocument(
                content=content,
                title=f"{drug_name} — {section_title} (FDA Label)",
                source="FDA Drug Label (openFDA)",
                url=url,
                metadata={
                    "drug_name": drug_name,
                    "section": field_key,
                    "manufacturer": manufacturer,
                    "api": "openfda_drug_label",
                },
            ))
        return docs


class ClinicalTrialsLoader:
    """
    Fetch study summaries from ClinicalTrials.gov using their free public API v2.
    No API key required.

    API docs: https://clinicaltrials.gov/data-api/api
    Endpoint:  https://clinicaltrials.gov/api/v2/studies?query.cond=<CONDITION>&...

    We extract the study's official title, brief summary, eligibility criteria, and
    primary outcomes — the most clinically informative fields for a RAG system — and
    wrap each study as a MedicalDocument.

    Usage example
    -------------
    loader = ClinicalTrialsLoader()
    docs = loader.search("type 2 diabetes GLP-1", max_results=5, status="COMPLETED")
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search(
        self,
        condition_or_query: str,
        max_results: int = 10,
        status: Optional[str] = "COMPLETED",
        min_enrollment: int = 50,
    ) -> List[MedicalDocument]:
        """
        Search ClinicalTrials.gov by condition, intervention, or free-text query.

        Parameters
        ----------
        condition_or_query : str
            Condition name, drug name, or free-text query.
        max_results : int
            Maximum studies to fetch (capped at 100 per API call).
        status : str or None
            Filter by study status. Common values: "COMPLETED", "RECRUITING",
            "ACTIVE_NOT_RECRUITING". Pass None to skip the filter.
        min_enrollment : int
            Skip studies with fewer participants than this (reduces noise from tiny studies).
        """
        params: Dict[str, Any] = {
            "query.cond": condition_or_query,
            "pageSize": min(max_results, 100),
            "format": "json",
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,BriefSummary,"
                "EligibilityCriteria,PrimaryOutcomeMeasure,"
                "OverallStatus,EnrollmentCount,StartDate,CompletionDate,"
                "Condition,InterventionName,LeadSponsorName"
            ),
        }
        if status:
            params["filter.overallStatus"] = status

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            studies = resp.json().get("studies", [])
            time.sleep(0.3)
        except Exception as e:
            print(f"ClinicalTrials.gov search error for '{condition_or_query}': {e}")
            return []

        docs = []
        for study in studies:
            doc = self._study_to_document(study, condition_or_query, min_enrollment)
            if doc:
                docs.append(doc)
        return docs

    def _study_to_document(
        self,
        study: dict,
        query: str,
        min_enrollment: int,
    ) -> Optional[MedicalDocument]:
        """Convert one ClinicalTrials.gov study JSON to a MedicalDocument."""
        # The v2 API nests fields under protocolSection
        ps = study.get("protocolSection", study)  # handle both flat and nested
        id_module = ps.get("identificationModule", {})
        desc_module = ps.get("descriptionModule", {})
        status_module = ps.get("statusModule", {})
        elig_module = ps.get("eligibilityModule", {})
        outcomes_module = ps.get("outcomesModule", {})
        design_module = ps.get("designModule", {})
        arms_module = ps.get("armsInterventionsModule", {})
        sponsor_module = ps.get("sponsorCollaboratorsModule", {})

        nct_id = id_module.get("nctId", "")
        title = id_module.get("officialTitle") or id_module.get("briefTitle") or nct_id
        brief_summary = desc_module.get("briefSummary", "").strip()

        # Enrollment filter
        enroll_info = design_module.get("enrollmentInfo", {})
        enrollment = enroll_info.get("count", 0) if isinstance(enroll_info, dict) else 0
        if enrollment and enrollment < min_enrollment:
            return None

        if not brief_summary or len(brief_summary) < 60:
            return None

        # Build a rich content block from multiple sections
        sections = [f"Study: {title}", f"Summary: {brief_summary}"]

        status = status_module.get("overallStatus", "")
        start = status_module.get("startDateStruct", {}).get("date") or status_module.get("startDate", "")
        end = status_module.get("completionDateStruct", {}).get("date") or status_module.get("completionDate", "")
        if status:
            sections.append(f"Status: {status}" + (f" (started {start}, ended {end})" if start else ""))

        conditions = ps.get("conditionsModule", {}).get("conditions", [])
        if conditions:
            sections.append("Conditions: " + "; ".join(conditions[:5]))

        interventions = [
            i.get("name", "") for i in arms_module.get("interventions", []) if i.get("name")
        ]
        if interventions:
            sections.append("Interventions: " + "; ".join(interventions[:5]))

        primary_outcomes = [
            o.get("measure", "") for o in outcomes_module.get("primaryOutcomes", []) if o.get("measure")
        ]
        if primary_outcomes:
            sections.append("Primary Outcomes: " + "; ".join(primary_outcomes[:3]))

        eligibility = elig_module.get("eligibilityCriteria", "")
        if eligibility:
            # Truncate overly long eligibility text
            sections.append("Eligibility: " + eligibility[:600] + ("..." if len(eligibility) > 600 else ""))

        sponsor = sponsor_module.get("leadSponsor", {}).get("name", "")
        if sponsor:
            sections.append(f"Lead sponsor: {sponsor}")

        content = "\n".join(sections)
        year = start[:4] if start and len(start) >= 4 else None

        return MedicalDocument(
            content=content,
            title=f"ClinicalTrial ({nct_id}): {title[:100]}",
            source="ClinicalTrials.gov",
            year=year,
            url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "https://clinicaltrials.gov",
            metadata={
                "nct_id": nct_id,
                "status": status,
                "enrollment": enrollment,
                "query": query,
                "api": "clinicaltrials_v2",
            },
        )
