# src/serialize.py
def serialize_record(job):
    """
    Chuyển 1 job dict thành (text, metadata)
    """
    title = job.get("name", "")
    company = job.get("company", "")
    locations = job.get("locations", [])
    description = job.get("description", "")
    requirements = job.get("requirements", "")
    skills = job.get("skills", job.get("skill", []))

    # text để embedding
    text_parts = [
        f"[TITLE] {title}",
        f"[COMPANY] {company} | [LOC] {', '.join(locations)}",
        f"[DESCRIPTION] {description}",
        f"[REQUIREMENTS] {requirements}",
        f"[SKILLS] {', '.join(skills)}"
    ]
    text = "\n".join(text_parts)

    # metadata để lưu trong vectordb
    meta = {
        "title": title,
        "company": company,
        "location": ", ".join(locations),
        "skills": skills
    }

    return text, meta
