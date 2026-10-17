import os

log_dir = r"C:\Users\Andrew\.gemini\antigravity\brain\969ccf87-2d21-4712-9db7-5cd578e6379d\.system_generated\tasks"
logs_to_read = ["task-3903.log", "task-3914.log", "task-3915.log", "task-3916.log"]
out_file = r"d:\belief-transformer\V3\codex_transcript.tex"

with open(out_file, "w", encoding="utf-8") as out:
    out.write("\\documentclass{article}\n")
    out.write("\\usepackage[margin=1in]{geometry}\n")
    out.write("\\usepackage{fvextra}\n")
    out.write("\\begin{document}\n")
    out.write("\\title{Codex Salvaged Transcript (Cleaned)}\n")
    out.write("\\date{}\n")
    out.write("\\maketitle\n")
    
    for log_name in logs_to_read:
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path): continue
        out.write(f"\\section*{{{log_name.replace('_', '\\_')}}}\n")
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("codex\n"):
                out.write("\\begin{Verbatim}[breaklines=true]\n")
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.startswith("exec\n") or next_line.startswith("collab:") or next_line.startswith("Wall time:"):
                        break
                    out.write(next_line)
                    i += 1
                out.write("\\end{Verbatim}\n")
                continue
            i += 1
            
    out.write("\\end{document}\n")
