import os
import sys

def list_c_drive_style():
    """
    Lists the current directory files in a format simulating Windows 'dir' command on C: drive.
    """
    path = "."
    abs_path = os.path.abspath(path)

    # Simulate C: drive header
    print(f" Volume in drive C is Windows")
    print(f" Volume Serial Number is A1B2-C3D4")
    print(f"")
    print(f" Directory of C:\\{os.path.basename(abs_path)}")
    print(f"")

    try:
        entries = os.listdir(path)
        file_count = 0
        dir_count = 0
        total_size = 0

        # Sort for consistency
        entries.sort()

        # Add . and .. simulation
        import datetime
        now = datetime.datetime.now()
        date_str = now.strftime("%m/%d/%Y  %I:%M %p")

        print(f"{date_str}    <DIR>          .")
        print(f"{date_str}    <DIR>          ..")
        dir_count += 2

        for entry in entries:
            full_path = os.path.join(path, entry)
            stats = os.stat(full_path)
            dt = datetime.datetime.fromtimestamp(stats.st_mtime)
            date_str = dt.strftime("%m/%d/%Y  %I:%M %p")

            if os.path.isdir(full_path):
                print(f"{date_str}    <DIR>          {entry}")
                dir_count += 1
            else:
                size = stats.st_size
                total_size += size
                # Format size with commas
                print(f"{date_str}    {size:>14,} {entry}")
                file_count += 1

        print(f"              {file_count} File(s)    {total_size:,} bytes")
        print(f"              {dir_count} Dir(s)     999,999,999 bytes free")

    except Exception as e:
        print(f"Error listing directory: {e}")

if __name__ == "__main__":
    list_c_drive_style()
