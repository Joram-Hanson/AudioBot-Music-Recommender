console.log("AudioBot frontend loaded");

let currentSeconds = 0;
let totalSeconds = 137; // default duration, matches "2:17"

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
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

// Fake dataset of tracks (each with its own "genre" and fake duration for demo purposes)
const realTracks = [
    { track_name: "Fast Car", artist: "Tracy Chapman", genre: "acoustic", duration: 296 },
    { track_name: "Talkin Bout a Revolution", artist: "Tracy Chapman", genre: "folk", duration: 143 },
    { track_name: "Mountains O Things", artist: "Tracy Chapman", genre: "acoustic", duration: 297 },
    { track_name: "For My Lover", artist: "Tracy Chapman", genre: "folk-rock", duration: 261 },
];

function playTrack(track) {
    // Update "now playing" info
    document.getElementById("now-playing-title").textContent = track.track_name;
    document.getElementById("now-playing-artist").textContent = track.artist;
    document.getElementById("now-playing-genre").textContent = track.genre;

    // Reset progress bar for the "new song"
    currentSeconds = 0;
    totalSeconds = track.duration;
    document.getElementById("total-time").textContent = formatTime(totalSeconds);
    document.getElementById("progress-fill").style.width = "0%";

    // Re-trigger the recommendation loading shimmer, as if fetching new recommendations
    loadRecommendations();
}

function loadRecommendations() {
    const list = document.getElementById("track-list");
    list.innerHTML = `
        <li class="loading-card"></li>
        <li class="loading-card"></li>
        <li class="loading-card"></li>
        <li class="loading-card"></li>
    `;

    setTimeout(() => {
        list.innerHTML = "";
        realTracks.forEach(track => {
            const li = document.createElement("li");
            li.className = "track-card";
            li.innerHTML = `
                <span class="track-name">${track.track_name}</span>
                <span class="artist-name">${track.artist}</span>
            `;
            li.addEventListener("click", () => playTrack(track));
            list.appendChild(li);
        });
    }, 1200);
}

loadRecommendations();