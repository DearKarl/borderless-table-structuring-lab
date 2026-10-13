"""Encode Docker bind mounts as one standards-compliant CSV record."""
import csv
import io

def bind_mount(source,target,readonly=True):
 fields=['type=bind','src='+str(source),'dst='+str(target)]
 if readonly:fields.append('readonly')
 stream=io.StringIO(newline='');csv.writer(stream,lineterminator='').writerow(fields)
 encoded=stream.getvalue();assert next(csv.reader([encoded]))==fields
 return encoded
