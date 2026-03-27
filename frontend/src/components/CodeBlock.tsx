import { Box } from "@mui/material";
import { Highlight, themes } from "prism-react-renderer";

interface Props {
  code: string;
  language?: string;
  startLine?: number;
}

export default function CodeBlock({
  code,
  language = "java",
  startLine = 1,
}: Props) {
  return (
    <Highlight theme={themes.nightOwl} code={code.trim()} language={language}>
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <Box
          component="pre"
          sx={{
            ...style,
            margin: 0,
            padding: 2,
            borderRadius: 2,
            overflow: "auto",
            fontSize: "0.82rem",
            lineHeight: 1.6,
            maxHeight: 400,
          }}
        >
          {tokens.map((line, i) => {
            const lineProps = getLineProps({
              line,
              key: i,
            });
            return (
              <Box
                key={i}
                {...lineProps}
                sx={{ display: "flex", "&:hover": { bgcolor: "rgba(255,255,255,0.04)" } }}
              >
                <Box
                  component="span"
                  sx={{
                    display: "inline-block",
                    width: 48,
                    textAlign: "right",
                    pr: 2,
                    color: "rgba(255,255,255,0.25)",
                    userSelect: "none",
                    flexShrink: 0,
                  }}
                >
                  {startLine + i}
                </Box>
                <span>
                  {line.map((token, key) => {
                    const tokenProps = getTokenProps({
                      token,
                      key,
                    });
                    return <span key={key} {...tokenProps} />;
                  })}
                </span>
              </Box>
            );
          })}
        </Box>
      )}
    </Highlight>
  );
}
