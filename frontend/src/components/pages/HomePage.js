import {Box, ButtonBase, styled} from "@mui/material";
import Typography from "@mui/material/Typography";
import {useNavigate} from "react-router-dom";
import data_upload from "../../assets/data-upload.png";
import use_existing_data from "../../assets/use-existing-data.png";
import {CategoriesPageLink} from "./CategoriesPage";
import {UploadPageLink} from "./UploadPage";

export const HomePageLink = "/";

export default function HomePage() {
    const navigate = useNavigate();

    const buttons = [
        {
            url: use_existing_data,
            title: "Use Existing Data",
            width: "50%",
            redirect: () => navigate(CategoriesPageLink, {state: {dataset: "default"}})
        },
        {
            url: data_upload,
            title: "Upload Data",
            width: "50%",
            redirect: () => navigate(UploadPageLink)
        }
    ];

    const ImageButton = styled(ButtonBase)(({theme}) => ({
        position: "relative",
        height: 400,
        [theme.breakpoints.down("sm")]: {
            width: "100% !important",
            height: 100
        },
        "&:hover, &.Mui-focusVisible": {
            zIndex: 1,
            "& .MuiImageBackdrop-root": {
                opacity: 0.15
            },
            "& .MuiImageMarked-root": {
                opacity: 0
            },
            "& .MuiTypography-root": {
                border: "4px solid currentColor"
            }
        }
    }));

    const ImageSrc = styled("span")({
        position: "absolute",
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
        backgroundSize: "cover",
        backgroundPosition: "center 40%"
    });

    const Image = styled("span")(({theme}) => ({
        position: "absolute",
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: theme.palette.common.white
    }));

    const ImageBackdrop = styled("span")(({theme}) => ({
        position: "absolute",
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
        backgroundColor: theme.palette.common.black,
        opacity: 0.4,
        transition: theme.transitions.create("opacity")
    }));

    const ImageMarked = styled("span")(({theme}) => ({
        height: 3,
        width: 18,
        backgroundColor: theme.palette.common.white,
        position: "absolute",
        bottom: -2,
        left: "calc(50% - 9px)",
        transition: theme.transitions.create("opacity")
    }));

    return (
        <Box sx={{display: "flex", flexWrap: "wrap", minWidth: 300, width: "100%"}}>
            {buttons.map((button) => (
                <ImageButton
                    focusRipple
                    onClick={button.redirect}
                    key={button.title}
                    style={{
                        width: button.width
                    }}
                >
                    <ImageSrc style={{backgroundImage: `url(${button.url})`}}/>
                    <ImageBackdrop className="MuiImageBackdrop-root"/>
                    <Image>
                        <Typography
                            component="span"
                            variant="subtitle1"
                            color="inherit"
                            sx={{
                                position: "relative",
                                p: 4,
                                pt: 2,
                                pb: (theme) => `calc(${theme.spacing(1)} + 6px)`
                            }}
                        >
                            {button.title}
                            <ImageMarked className="MuiImageMarked-root"/>
                        </Typography>
                    </Image>
                </ImageButton>
            ))}
        </Box>
    );
}
