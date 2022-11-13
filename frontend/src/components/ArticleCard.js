import {ReadMore} from "@mui/icons-material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import CardHeader from "@mui/material/CardHeader";
import CardMedia from "@mui/material/CardMedia";
import Collapse from "@mui/material/Collapse";
import IconButton, {IconButtonProps} from "@mui/material/IconButton";
import {styled} from "@mui/material/styles";
import Typography from "@mui/material/Typography";
import * as React from "react";
import {useNavigate} from "react-router-dom";
import placeholder from "../assets/placeholder.png";
import {SummaryPageLink} from "./pages/SummaryPage";

export default function ArticleCard({article}) {
    const navigate = useNavigate();

    const [expanded, setExpanded] = React.useState(false);

    interface ExpandMoreProps extends IconButtonProps {
        expand: boolean;
    }

    const ExpandMore = styled((props: ExpandMoreProps) => {
        const {expand, ...other} = props;
        return <IconButton {...other} />;
    })(({theme, expand}) => ({
        transform: !expand ? "rotate(0deg)" : "rotate(180deg)",
        marginLeft: "auto",
        transition: theme.transitions.create("transform", {
            duration: theme.transitions.duration.shortest
        })
    }));

    const handleExpandClick = () => {
        setExpanded(!expanded);
    };

    return (
        <Card sx={{maxWidth: 350, minWidth: 350, minHeight: 350}}>
            <CardHeader
                title={article.title}
            />
            <CardMedia
                component="img"
                height="194"
                image={"image" in article ? "" : placeholder}
                src={"image" in article ? `data:image/png;base64, ${article.image}` : ""}
                alt=""
            />
            <CardContent>
                {""}
            </CardContent>
            <CardActions disableSpacing>
                <IconButton aria-label="go to article"
                            onClick={() => navigate(SummaryPageLink, {state: {article: article}})}>
                    <ReadMore/>
                </IconButton>
                <ExpandMore
                    expand={expanded}
                    onClick={handleExpandClick}
                    aria-expanded={expanded}
                    aria-label="show summary"
                >
                    <ExpandMoreIcon/>
                </ExpandMore>
            </CardActions>
            <Collapse in={expanded} timeout="auto" unmountOnExit>
                <CardContent>
                    <Typography paragraph>{"summary" in article ? article.summary : ""}</Typography>
                </CardContent>
            </Collapse>
        </Card>
    );
}