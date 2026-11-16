import pypdf
import os
import pptx
import json
import embed_utils

#prevents duplicate slides
def remove_duplicates(entries):
    kept = []
    for i, entry in enumerate(entries):
        if i + 1 < len(entries):
            this_text = entry["text"]
            next_text = entries[i + 1]["text"]
            added_at_end = next_text.startswith(this_text)
            all_lines_in_next = set(this_text.splitlines()) <= set(next_text.splitlines())
            if added_at_end or all_lines_in_next:
                continue
        kept.append(entry)
    return kept

#returns one entry per page in the pdf file
def extract_from_pdf(file_path):
    pdf = pypdf.PdfReader(file_path)
    entries = [] #stores one entry per page
    for i, page in enumerate(pdf.pages, start =1):
        page_text = page.extract_text().strip() #extracts the text from the page and removes extra whitespace
        if page_text: #skips pages with no text, like image-only pages
            entries.append( {
                "source": os.path.basename(file_path),
                "location": f"page {i}",
                "text": page_text
            }) #stores the page as a dictionary with its file name, page number and text
    return remove_duplicates(entries)


#returns one entry per slide in the pptx file
def extract_from_pptx(file_path):
    presentation = pptx.Presentation(file_path)
    entries = [] #stores one entry per slide
    for i, slide in enumerate(presentation.slides, start =1):
        slide_text = "\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text")).strip() #joins the text from every shape on the slide
        if slide_text: #skips slides with no text
            entries.append( {
                "source": os.path.basename(file_path),
                "location": f"slide {i}",
                "text": slide_text
            }) #stores the slide as a dictionary with its file name, slide number and text
    return remove_duplicates(entries)


def extract_all(folder):
    all_entries = [] #stores the entries from every file in one list

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        extension = os.path.splitext(name)[1].lower() #gets the extension

        if extension == ".pdf":
            entries = extract_from_pdf(path)
        elif extension == ".pptx":
            entries = extract_from_pptx(path)
        else:
            print(f"Skipping {name}: not a pdf or pptx")
            continue

        all_entries.extend(entries) #adds this file's entries to the big list

    return all_entries



def main():
    folder = "notes"  # Specify the folder containing the files
    entries = extract_all(folder)
    texts = [entry["text"] for entry in entries]  # Extract the text from each entry
    vectors = embed_utils.embed(texts)  # Get the vector representations of the texts
    #zips the entires and vectors together and adds the vector to each entry as a list for JSON serialization
    for entry, vector in zip(entries, vectors):
        entry["embedding"] = vector.tolist()  # Convert the numpy array to a list for JSON serialization

    with open("index.json", "w", encoding="utf-8") as f:
        json.dump(entries, f)  # Write the data to a JSON file


if __name__ == "__main__":
    main()
