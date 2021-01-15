$(function () {
    let msg = "";
    let bot_msg;
    let idx = 0;

    $("#chat-submit").click(function (e) {
        e.preventDefault();
        msg = $("#chat-input__text").val();
        if (msg.trim() == '') {
            return false;
        }
        generate_message(msg, '', 'self');
        getNeuralResponse(msg);
    })

    function generate_message(msg, link, type) {
        idx++;
        let str = "";
        if (type == 'self') {
            str += "<div id='cm-msg-" + idx + "' class=\"chat-msg " + type + "\">";
            str += "<div class=\"cm-msg-text\">";
            str += msg;
            str += "<\/div>";
            str += "<\/div>";

        } else {
            str += "<div id='cm-msg-" + idx + "' class=\"chat-msg " + type + "\">";
            str += "<\/span>";
            str += "<div class=\"cm-msg-text\">";
            str += msg
            if(link != '')
            {
                str += "<a href=\""
                str += link;
                str += "\">" + 'здесь'+ ".<\/a>"
            }
            str += "<\/div>";
            str += "<\/div>";
        }

        $(".chat-logs").append(str);
        $("#cm-msg-" + idx).hide().fadeIn(500);

        if (type == 'self') {
            $("#chat-input__text").val('');
        }

        $(".chat-logs").stop().animate({
            scrollTop: $(".chat-logs")[0].scrollHeight
        }, 1000);
    }

    function getNeuralResponse(msg) {
        let rawText = msg;
        $.get("/get", { msg: rawText }).done(function(data) {
            let botMsg = data.split('\n');
            console.log(botMsg);
            generate_message(botMsg[1], botMsg[0], 'bot');
        });
    }

    $("#chat-click").click(function () {
        $("#chat-click").hide('scale');
        $(".chat-box").show('scale');
        $(".chat-box-welcome__header").show('scale');
        $("#chat-box__wraper").show();
    })

    $(".chat-box-toggle").click(function () {
        $("#chat-click").show('scale');
        $(".chat-box").hide('scale');
        $(".chat-box-welcome__header").hide('scale');
        $("#chat-box__wraper").hide('scale');
    })
})