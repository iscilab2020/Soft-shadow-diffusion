#!/bin/bash

# Check if at least four directory paths are provided as arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 /path/to/npy_file /path/to/output [delete] [gif] [video]"
    exit 1
fi

npy_file="$1"
xml_dir="$2"
exr_dir="$xml_dir"
output_dir="$xml_dir"
delete_pngs="$3"
create_gif="$4"
process_video="$5"

max_jobs=16
jobs=0

# Function to process .npy file and generate XML files
process_npy_file() {
    npy_file="$1"
    xml_output_dir="$2"

    mkdir -p "$xml_output_dir"
    python generate_xml.py "$npy_file" "$xml_output_dir"
    echo "XML files generated in $xml_output_dir"
}

process_xml() {
    for xml_file in "$xml_dir"/*.xml; do
        [ -f "$xml_file" ] || continue

        exr_file="$(basename "$xml_file" .xml).exr"

        exr_file="${output_dir}/${exr_file}"


        echo "Output directory: $output_dir"
        echo "XML file: $xml_file"
        echo "EXR file: $exr_file"


        if [ -f "$exr_file" ]; then
            echo "EXR file already exists for $xml_file, skipping..."
            continue
        fi


        if mitsuba "$xml_file" -o "$exr_file" & then
            echo "Started converting $xml_file to $exr_file"
        else
            echo "Failed to start conversion for $xml_file" >&2
            continue
        fi

        ((jobs++))
        if ((jobs >= max_jobs)); then
            wait -n
            ((jobs--))
        fi
    done
    wait
    echo "XML to EXR conversion completed."
}


# Function to process EXR files
process_exr() {
    for exr_file in "$exr_dir"/*.exr; do
        [ -f "$exr_file" ] || continue
        png_file="$(basename "$exr_file" .exr).png"
        png_file="${output_dir}/${png_file}"

        if [ -f "$png_file" ]; then
            echo "PNG file already exists for $xml_file, skipping..."
            continue
        fi
        convert "$exr_file" "$png_file"
        echo "Converted $exr_file to $png_file"
    done
}

# Function to create animated GIF and optionally convert to video
#!/bin/bash
# Function to create animated GIF and optionally convert to video
create_gif_and_video() {
    if [ "$create_gif" = "gif" ]; then
        convert -delay 30 $(ls "$output_dir"/*.png | sort -V) \( -clone -1 -set delay 300 \) -loop 0 "$output_dir/output.gif"
        if [ $? -eq 0 ]; then
            echo "Animated GIF created successfully."
        else
            echo "Failed to create animated GIF." >&2
        fi
    fi

    if [ "$process_video" = "video" ]; then
        ffmpeg -framerate 24 -i "$output_dir/%d.png" "$output_dir/output.mp4"
        echo "Video created successfully."
    fi

    if [ "$delete_pngs" = "delete" ]; then
        rm -f "$output_dir"/*.png
        echo "PNG files deleted."
    fi
}


# Main Processing Steps
process_npy_file "$npy_file" "$xml_dir"
process_xml
process_exr
create_gif_and_video
