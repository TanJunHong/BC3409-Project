import {Grid, Typography} from "@mui/material";
import {useLocation} from "react-router-dom";
import placeholder from "../../assets/placeholder.png";

export const SummaryPageLink = "/summary";

export default function SummaryPage() {
    const {state} = useLocation();
    const article = state.article;
    return (
        <Grid container spacing={6} justifyContent={"space-evenly"} alignItems={"stretch"}>
            <Grid item xs={12}>
                <Typography variant="h3">{article["title"]}</Typography>
                <a href={article["url"]} style={{fontSize: "0.75em", color: "#ffffcc"}}>{article["url"]}</a>
                <br/>
                <br/>
                <Typography variant="h4">Summary</Typography>
                <Typography variant="subtitle1">{article["summary"]}</Typography>
                <br/>
                <br/>
                <Typography variant="h5">Article Text</Typography>
                <img src={"image" in article ? `data:image/png;base64, ${article.image}` : placeholder}
                     style={{objectFit: "contain"}}
                     alt=""></img>
                <Typography paragraph={true} variant="body1" align="left">{article["text"]}</Typography>
            </Grid>
        </Grid>
    );
}
