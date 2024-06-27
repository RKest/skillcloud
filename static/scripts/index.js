var interval = null;

document.getElementById("measure").addEventListener("click", async () => {
    const main = document.getElementById("main")
    measurements = {};
    document.querySelectorAll("div").forEach((el) => {
        measurements[el.innerHTML.trim()] = {
            "x": el.offsetWidth / main.offsetWidth,
            "y": el.offsetHeight / main.offsetHeight,
        };
    })
    await fetch("/measure", {
        method: "POST",
        body: JSON.stringify(measurements),
        headers: {"Content-Type": "application/json"}
    }).catch(console.error);
})

document.getElementById("beginPolling").addEventListener("click", async () => {
    interval = setInterval(() => htmx.trigger("main", "startPolling"), 300);
});

document.getElementById("endPolling").addEventListener("click", () => {
    clearInterval(interval);
    interval = null;
});
