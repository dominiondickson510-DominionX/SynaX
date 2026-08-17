// services/citation/citation_service.js
const express = require("express");
const { Cite } = require("@citation-js/core");

require("@citation-js/plugin-csl");

const app = express();

app.use(express.json({ limit: "4mb" }));

app.post("/format", (req, res) => {
    try {
        const {
            items,
            style = "apa",
            output_type = "bibliography",
            format = "text",
            lang = "en-US",
        } = req.body;

        if (!Array.isArray(items) || items.length === 0) {
            return res.status(400).json({
                error: "items must be a non-empty array",
            });
        }

        const cite = new Cite(items);

        const result = cite.format(output_type, {
            template: style,
            lang,
            format,
        });

        res.json({ result });
    } catch (error) {
        console.error("[Citation.js]", error);
        res.status(500).json({ error: error.message });
    }
});

app.get("/health", (req, res) => {
    res.json({
        status: "ok",
        service: "synax-citation",
    });
});

const PORT = 3100;

app.listen(PORT, "127.0.0.1", () => {
    console.log(
        `Citation service listening on http://127.0.0.1:${PORT}`
    );
});