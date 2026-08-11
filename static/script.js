console.log("AudioBot frontend loaded");

let currentSeconds = 0;
let totalSeconds = 137;

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
}

function tickProgress() {
    currentSeconds += 1;
    if (currentSeconds > totalSeconds) currentSeconds = 0;
    const percent = (currentSeconds / totalSeconds) * 100;
    document.getElementById("progress-fill").style.width = percent + "%";
    document.getElementById("current-time").textContent = formatTime(currentSeconds);
}

setInterval(tickProgress, 1000);

async function loadGenres() {
    const response = await fetch("/genres");
    const genres = await response.json();

    const genreSelect = document.getElementById("genre-select");
    genres.forEach(genre => {
        const option = document.createElement("option");
        option.value = genre;
        option.textContent = genre;
        genreSelect.appendChild(option);
    });
}

async function loadSongsForGenre(genre) {
    const songSelect = document.getElementById("song-select");
    songSelect.innerHTML = `<option value="">-- Loading songs... --</option>`;
    songSelect.disabled = true;

    const response = await fetch(`/songs/${encodeURIComponent(genre)}`);
    const songs = await response.json();

    songSelect.innerHTML = `<option value="">-- Select a song --</option>`;
    songs.forEach(song => {
        const option = document.createElement("option");
        option.value = song.track_id;
        option.textContent = `${song.track_name} - ${song.artists}`;
        songSelect.appendChild(option);
    });
    songSelect.disabled = false;
}

async function playTrack(trackId) {
    const list = document.getElementById("track-list");
    list.innerHTML = `
        <li class="loading-card"></li>
        <li class="loading-card"></li>
        <li class="loading-card"></li>
        <li class="loading-card"></li>
    `;

    try {
        const response = await fetch(`/recommend/${trackId}`);
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        document.getElementById("now-playing-title").textContent = data.input_track;
        document.getElementById("now-playing-artist").textContent = "";
        document.getElementById("now-playing-genre").textContent = data.actual_genre;

        currentSeconds = 0;
        document.getElementById("progress-fill").style.width = "0%";

        list.innerHTML = "";
        data.recommendations.forEach(track => {
            const li = document.createElement("li");
            li.className = "track-card";
            li.innerHTML = `
                <span class="track-name">${track.track_name}</span>
                <span class="artist-name">${track.artists}</span>
            `;
            li.addEventListener("click", () => playTrack(track.track_id));
            list.appendChild(li);
        });
    } catch (err) {
        console.error("Failed to fetch recommendations:", err);
    }
}

document.getElementById("genre-select").addEventListener("change", (e) => {
    if (e.target.value) {
        loadSongsForGenre(e.target.value);
    }
});

document.getElementById("song-select").addEventListener("change", (e) => {
    if (e.target.value) {
        playTrack(e.target.value);
    }
});

loadGenres();